"""Minimal passwordless session handling and anonymous question quota."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import urllib.parse
import uuid
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from backend.config import get_settings

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])
FREE_QUESTION_LIMIT = 2
ANONYMOUS_COOKIE = "rag_free_questions"
SESSION_COOKIE = "rag_session"
USER_COOKIE = "rag_user_id"
OAUTH_STATE_COOKIE = "rag_oauth_state"


class LoginRequest(BaseModel):
    email: str


def _sign(value: str) -> str:
    secret = get_settings().auth_secret.encode()
    encoded = base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")
    signature = hmac.new(secret, encoded.encode(), hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


def _read_signed(value: Optional[str]) -> Optional[str]:
    if not value or "." not in value:
        return None
    encoded, signature = value.rsplit(".", 1)
    expected = hmac.new(get_settings().auth_secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        return base64.urlsafe_b64decode(encoded + "===").decode()
    except (ValueError, UnicodeDecodeError):
        return None


async def get_supabase_user(request: Request) -> Optional[dict]:
    settings = get_settings()
    authorization = request.headers.get("Authorization", "")
    if not settings.supabase_url or not settings.supabase_anon_key or not authorization.startswith("Bearer "):
        return None
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(
                f"{settings.supabase_url.rstrip('/')}/auth/v1/user",
                headers={"Authorization": authorization, "apikey": settings.supabase_anon_key},
            )
        return response.json() if response.is_success else None
    except httpx.HTTPError:
        return None


async def is_authenticated(request: Request) -> bool:
    return await get_supabase_user(request) is not None or get_session(request) is not None


def get_session(request: Request) -> Optional[dict]:
    payload = _read_signed(request.cookies.get(SESSION_COOKIE))
    if not payload:
        return None
    try:
        session = json.loads(payload)
        return session if session.get("email") else None
    except (json.JSONDecodeError, AttributeError):
        return None


async def get_user_key(request: Request, response: Optional[Response] = None) -> str:
    """Return a stable owner key for per-user data."""
    supabase_user = await get_supabase_user(request)
    if supabase_user:
        identity = supabase_user.get("id") or supabase_user.get("email")
        if identity:
            return f"supabase:{identity}"

    session = get_session(request)
    if session:
        return f"session:{session['email'].strip().lower()}"

    anonymous_id = _read_signed(request.cookies.get(USER_COOKIE))
    if not anonymous_id:
        anonymous_id = str(uuid.uuid4())
        if response is not None:
            response.set_cookie(
                USER_COOKIE,
                _sign(anonymous_id),
                max_age=60 * 60 * 24 * 365,
                httponly=True,
                samesite="lax",
            )
    return f"anonymous:{anonymous_id}"


def _provider_config(provider: str) -> tuple[str, str, str, str, str]:
    settings = get_settings()
    if provider == "google":
        if not settings.google_client_id or not settings.google_client_secret:
            raise HTTPException(status_code=503, detail="Google login is not configured.")
        return (
            settings.google_client_id,
            settings.google_client_secret.get_secret_value(),
            "https://accounts.google.com/o/oauth2/v2/auth",
            "https://oauth2.googleapis.com/token",
            "https://www.googleapis.com/oauth2/v3/userinfo",
        )
    if provider == "github":
        if not settings.github_client_id or not settings.github_client_secret:
            raise HTTPException(status_code=503, detail="GitHub login is not configured.")
        return (
            settings.github_client_id,
            settings.github_client_secret.get_secret_value(),
            "https://github.com/login/oauth/authorize",
            "https://github.com/login/oauth/access_token",
            "https://api.github.com/user",
        )
    raise HTTPException(status_code=404, detail="Unknown login provider.")


def _callback_url(request: Request, provider: str) -> str:
    return str(request.base_url).rstrip("/") + f"/api/v1/auth/{provider}/callback"


@router.get("/{provider}/login")
async def oauth_login(provider: str, request: Request, response: Response):
    client_id, _, authorization_url, _, _ = _provider_config(provider)
    state = secrets.token_urlsafe(32)
    response = RedirectResponse(
        authorization_url
        + "?"
        + urllib.parse.urlencode(
            {
                "client_id": client_id,
                "redirect_uri": _callback_url(request, provider),
                "response_type": "code",
                "scope": "openid email profile" if provider == "google" else "read:user user:email",
                "state": state,
            }
        )
    )
    response.set_cookie(OAUTH_STATE_COOKIE, _sign(state), max_age=600, httponly=True, samesite="lax")
    return response


@router.get("/{provider}/callback")
async def oauth_callback(provider: str, code: str, state: str, request: Request):
    client_id, client_secret, _, token_url, profile_url = _provider_config(provider)
    saved_state = _read_signed(request.cookies.get(OAUTH_STATE_COOKIE))
    if not saved_state or not hmac.compare_digest(saved_state, state):
        raise HTTPException(status_code=400, detail="Invalid OAuth state.")

    async with httpx.AsyncClient(timeout=15) as client:
        token_response = await client.post(
            token_url,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": _callback_url(request, provider),
            },
            headers={"Accept": "application/json"},
        )
        token_response.raise_for_status()
        access_token = token_response.json().get("access_token")
        if not access_token:
            raise HTTPException(status_code=400, detail="OAuth provider did not return an access token.")
        profile_response = await client.get(
            profile_url,
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
        )
        profile_response.raise_for_status()
        profile = profile_response.json()

        email = profile.get("email")
        if provider == "github" and not email:
            emails_response = await client.get("https://api.github.com/user/emails", headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"})
            emails_response.raise_for_status()
            email = next((item["email"] for item in emails_response.json() if item.get("primary") and item.get("verified")), None)
    if not email:
        raise HTTPException(status_code=400, detail="No verified email was returned by the provider.")

    redirect = RedirectResponse(get_settings().auth_frontend_url, status_code=303)
    session = json.dumps({"email": email, "provider": provider}, separators=(",", ":"))
    redirect.set_cookie(SESSION_COOKIE, _sign(session), max_age=60 * 60 * 24 * 30, httponly=True, samesite="lax")
    redirect.delete_cookie(OAUTH_STATE_COOKIE)
    return redirect


async def enforce_question_access(request: Request, response: Response) -> None:
    """Allow two anonymous questions, then require a signed session cookie."""
    if getattr(request.state, "question_access_checked", False):
        return
    request.state.question_access_checked = True
    if await is_authenticated(request):
        return

    try:
        used = int(_read_signed(request.cookies.get(ANONYMOUS_COOKIE)) or "0")
    except ValueError:
        used = 0
    if used >= FREE_QUESTION_LIMIT:
        raise HTTPException(
            status_code=401,
            detail="LOGIN_REQUIRED: Please log in to continue asking questions.",
            headers={"WWW-Authenticate": "Login"},
        )
    response.set_cookie(
        ANONYMOUS_COOKIE,
        _sign(str(used + 1)),
        max_age=60 * 60 * 24 * 30,
        httponly=True,
        samesite="lax",
    )


@router.post("/login")
async def login(credentials: LoginRequest, response: Response):
    if "@" not in credentials.email or not credentials.email.strip():
        raise HTTPException(status_code=422, detail="Please provide a valid email address.")
    session = json.dumps({"email": str(credentials.email)}, separators=(",", ":"))
    response.set_cookie(
        SESSION_COOKIE,
        _sign(session),
        max_age=60 * 60 * 24 * 30,
        httponly=True,
        samesite="lax",
    )
    return {"authenticated": True, "email": str(credentials.email)}


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(SESSION_COOKIE)
    return {"authenticated": False}


@router.get("/me")
async def me(request: Request):
    supabase_user = await get_supabase_user(request)
    session = get_session(request)
    user = supabase_user or session
    return {
        "authenticated": user is not None,
        "email": user.get("email") if user else None,
        "provider": "supabase" if supabase_user else (session.get("provider") if session else None),
    }


@router.get("/config")
async def auth_config():
    settings = get_settings()
    return {"supabase_url": settings.supabase_url, "supabase_anon_key": settings.supabase_anon_key}