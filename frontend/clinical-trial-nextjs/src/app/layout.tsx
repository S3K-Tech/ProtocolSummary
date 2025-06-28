import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Image from "next/image";
import Link from "next/link";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Clinical Trial Protocol",
  description: "AI-powered clinical protocol generator",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased bg-[#fafbfc] min-h-screen flex flex-col`}>
        {/* Header */}
        <header className="flex justify-between items-center px-8 py-2 bg-white border-b border-gray-200">
          <div className="flex items-center gap-4">
            <Image src="/s3ktech-logo.png" alt="S3KTech.ai" width={160} height={50} priority />
          </div>
          <nav className="flex gap-4">
            <Link href="#" className="text-gray-800 text-sm px-3 py-1 rounded hover:bg-gray-100">AI Platforms</Link>
            <Link href="#" className="text-gray-800 text-sm px-3 py-1 rounded hover:bg-gray-100">AI POCs</Link>
            <Link href="#" className="text-gray-800 text-sm px-3 py-1 rounded hover:bg-gray-100">AI Trainings</Link>
            <Link href="#" className="text-gray-800 text-sm px-3 py-1 rounded hover:bg-gray-100">About Us</Link>
            <Link href="#" className="text-blue-700 font-semibold text-sm px-3 py-1 rounded hover:bg-blue-50">Demo &gt;&gt;</Link>
          </nav>
        </header>
        {/* Main Layout */}
        <main className="flex flex-1 min-h-0">
          {/* Sidebar placeholder */}
          <aside className="bg-[#f3f3f3] w-[260px] border-r border-gray-200 hidden md:flex flex-col"></aside>
          <section className="flex-1 px-6 py-8">{children}</section>
        </main>
        {/* Footer */}
        <footer className="bg-black text-white text-center py-3 text-sm mt-auto">
          © 2025 S3K Technologies | All rights reserved | Designed & Developed By Webtactic
        </footer>
      </body>
    </html>
  );
}
