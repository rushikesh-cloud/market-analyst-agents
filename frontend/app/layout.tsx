import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Market Analyst",
  description: "Agent dashboard for market analysis",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
