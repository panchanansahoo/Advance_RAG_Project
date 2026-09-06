import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Advanced Multimodal Agentic RAG",
  description: "Evidence-Grounded Multi-Document Question Answering",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <nav className="fixed top-0 w-full z-50 glass-panel !rounded-none !border-t-0 !border-x-0 bg-slate-900/80 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white font-bold shadow-[0_0_15px_rgba(37,99,235,0.5)]">
              R
            </div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
              Agentic RAG
            </h1>
          </div>
          <div className="flex gap-6 text-sm font-medium">
            <Link href="/" className="text-slate-300 hover:text-white transition-colors">Chat</Link>
            <Link href="/dashboard" className="text-slate-300 hover:text-white transition-colors">Dashboard</Link>
          </div>
        </nav>
        <main className="pt-20 min-h-screen">
          {children}
        </main>
      </body>
    </html>
  );
}
