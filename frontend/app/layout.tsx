import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Market Analyst Dashboard",
  description: "Frontend for web, technical, fundamental, supervisor, and ingestion agents.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
