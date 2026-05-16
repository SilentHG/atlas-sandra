import "./styles.css";

export const metadata = {
  title: "ATLAS Trading Dashboard",
  description: "Live portfolio, positions, and strategy scoreboard for ATLAS"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
