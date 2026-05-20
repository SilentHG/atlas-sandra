import Link from "next/link";
import { Activity, Bot, Gauge, LineChart, ShieldAlert } from "lucide-react";
import "./globals.css";

const nav = [
  { href: "/", label: "Overview", icon: Activity },
  { href: "/strategies", label: "Strategies", icon: LineChart },
  { href: "/trading", label: "Trading", icon: Gauge },
  { href: "/risk", label: "Risk", icon: ShieldAlert },
  { href: "/agents", label: "Agents", icon: Bot }
];

export const metadata = {
  title: "ATLAS Trading Dashboard",
  description: "Live ATLAS trading operations dashboard"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen bg-atlas-bg text-atlas-text">
          <aside className="fixed inset-y-0 left-0 hidden w-64 border-r border-atlas-line bg-atlas-panel px-4 py-5 lg:block">
            <div className="mb-8">
              <div className="text-2xl font-semibold tracking-wide">ATLAS</div>
              <div className="mt-1 text-sm text-atlas-muted">Shah Equity Holdings</div>
            </div>
            <nav className="space-y-1">
              {nav.map((item) => {
                const Icon = item.icon;
                return (
                  <Link key={item.href} href={item.href} className="flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium text-atlas-muted transition hover:bg-atlas-panel2 hover:text-atlas-text">
                    <Icon size={18} />
                    {item.label}
                  </Link>
                );
              })}
            </nav>
          </aside>
          <div className="lg:pl-64">
            <header className="sticky top-0 z-20 border-b border-atlas-line bg-atlas-bg/95 px-4 py-3 backdrop-blur lg:hidden">
              <div className="mb-3 text-xl font-semibold">ATLAS</div>
              <nav className="grid grid-cols-5 gap-1 text-xs text-atlas-muted">
                {nav.map((item) => (
                  <Link key={item.href} href={item.href} className="rounded-md bg-atlas-panel px-2 py-2 text-center">
                    {item.label}
                  </Link>
                ))}
              </nav>
            </header>
            <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">{children}</main>
          </div>
        </div>
      </body>
    </html>
  );
}
