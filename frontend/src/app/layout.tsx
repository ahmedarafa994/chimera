import type { Metadata, Viewport } from "next";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";
import Script from "next/script";

// Use unified provider to eliminate redundant API calls
import QueryProvider from "@/providers/query-provider";
import { UnifiedModelProvider } from "@/providers/unified-model-provider";
import ProviderSyncProvider from "@/providers/provider-sync-provider";
import ResourceOptimizationProvider from "@/components/providers/ResourceOptimizationProvider";
import { WebVitalsTracker } from "@/components/WebVitalsTracker";
import { AuthProvider } from "@/providers/AuthProvider";

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#000000",
};

export const metadata: Metadata = {
  title: "Chimera Fuzzing Platform",
  description: "Advanced LLM Fuzzing and Security Testing",
  keywords: "LLM, fuzzing, security testing, AI, jailbreak, red team",
  authors: [{ name: "Chimera Team" }],
  robots: "index, follow",
  manifest: "/manifest.json",
  icons: {
    icon: [
      { url: "/favicon.ico", sizes: "48x48" },
      { url: "/favicon-16x16.svg", sizes: "16x16", type: "image/svg+xml" },
      { url: "/favicon-32x32.svg", sizes: "32x32", type: "image/svg+xml" },
    ],
    apple: "/icon-192x192.svg",
  },
  openGraph: {
    title: "Chimera Fuzzing Platform",
    description: "Advanced LLM Fuzzing and Security Testing",
    type: "website",
    url: "https://chimera.ai",
  },
};

import ErrorBoundary from "@/components/common/ErrorBoundary";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <head>
        {/* Resource hints for performance - preconnect to font origins */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />

        {/* Load only the fonts we need - removed duplicate preload to avoid "preloaded but not used" warnings */}
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />

        {/* DNS prefetch for external resources */}
        <link rel="dns-prefetch" href="//api.chimera.ai" />
      </head>
      <body
        className="font-sans antialiased"
        suppressHydrationWarning
      >
        <QueryProvider>
          <AuthProvider>
            <ProviderSyncProvider>
              <UnifiedModelProvider>
                <ResourceOptimizationProvider>
                  <ErrorBoundary>
                    {children}
                  </ErrorBoundary>
                  <Toaster />
                </ResourceOptimizationProvider>
              </UnifiedModelProvider>
            </ProviderSyncProvider>
          </AuthProvider>
        </QueryProvider>

        {/* Web Vitals tracking */}
        <WebVitalsTracker />

        {/* Service Worker registration */}
        <Script
          id="sw-registration"
          strategy="lazyOnload"
          dangerouslySetInnerHTML={{
            __html: `
              if ('serviceWorker' in navigator && '${process.env.NODE_ENV}' === 'production') {
                navigator.serviceWorker.register('/sw.js')
                  .then(registration => console.log('SW registered:', registration))
                  .catch(error => console.log('SW registration failed:', error));
              }
            `,
          }}
        />
      </body>
    </html>
  );
}
