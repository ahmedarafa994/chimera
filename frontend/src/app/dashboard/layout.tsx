"use client";

import { useState, useEffect } from "react";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import { KeyboardShortcuts } from "@/components/keyboard-shortcuts";

// Consistent className for the sidebar wrapper
const SIDEBAR_WRAPPER_CLASS = "hidden md:flex md:flex-col w-64 h-screen border-r overflow-hidden";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Use client-side only rendering to prevent hydration mismatches
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- Legitimate use case for hydration fix
    setMounted(true);
  }, []);

  // During SSR and initial hydration, render a minimal skeleton
  // that matches what the server renders
  if (!mounted) {
    return (
      <div className="flex h-screen overflow-hidden">
        <div className={SIDEBAR_WRAPPER_CLASS} />
        <div className="flex flex-1 flex-col overflow-hidden">
          <header className="sticky top-0 z-30 flex h-14 items-center border-b bg-background px-6" />
          <main className="flex-1 overflow-auto bg-background p-4 md:p-6">
            {children}
          </main>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <div className={SIDEBAR_WRAPPER_CLASS}>
        <Sidebar />
      </div>
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-auto bg-background p-4 md:p-6">
          {children}
        </main>
      </div>
      <KeyboardShortcuts />
    </div>
  );
}
