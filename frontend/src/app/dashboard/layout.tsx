"use client";

import { useState, useEffect } from "react";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import { KeyboardShortcuts } from "@/components/keyboard-shortcuts";
import { ProtectedRoute } from "@/components/auth/ProtectedRoute";

// Consistent className for the sidebar wrapper
const SIDEBAR_WRAPPER_CLASS = "hidden md:flex md:flex-col w-64 h-screen border-r overflow-hidden";

// Loading skeleton for the dashboard layout
function DashboardSkeleton({ children }: { children?: React.ReactNode }) {
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

// Dashboard content wrapper that renders after hydration
function DashboardContent({ children }: { children: React.ReactNode }) {
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
    return <DashboardSkeleton>{children}</DashboardSkeleton>;
  }

  // Wrap the dashboard content in ProtectedRoute for authentication
  return (
    <ProtectedRoute
      loadingMessage="Loading dashboard..."
      loadingComponent={<DashboardSkeleton />}
    >
      <DashboardContent>{children}</DashboardContent>
    </ProtectedRoute>
  );
}
