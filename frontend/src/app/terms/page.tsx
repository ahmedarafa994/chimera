"use client";

/**
 * Terms of Service Page
 *
 * @module app/terms/page
 */

import Link from "next/link";
import { ArrowLeft, ShieldAlert, FileText } from "lucide-react";
import {
    Card,
    CardContent,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";

export default function TermsPage() {
    return (
        <div className="min-h-screen bg-background text-foreground">
            {/* Background */}
            <div className="fixed inset-0 pointer-events-none overflow-hidden">
                <div className="absolute top-1/4 -left-32 w-96 h-96 bg-primary/10 rounded-full blur-[100px]" />
                <div className="absolute bottom-1/4 -right-32 w-96 h-96 bg-purple-500/10 rounded-full blur-[100px]" />
            </div>

            {/* Header */}
            <header className="relative z-10 p-4 border-b border-white/5">
                <div className="max-w-4xl mx-auto flex items-center justify-between">
                    <Link
                        href="/"
                        className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
                    >
                        <ArrowLeft className="w-4 h-4" />
                        <span className="text-sm font-medium">Back to Home</span>
                    </Link>

                    <Link href="/" className="flex items-center gap-2">
                        <ShieldAlert className="w-6 h-6 text-primary" />
                        <span className="text-lg font-bold bg-gradient-to-r from-primary to-purple-400 bg-clip-text text-transparent">
                            Chimera
                        </span>
                    </Link>

                    <div className="w-[100px]" />
                </div>
            </header>

            {/* Content */}
            <main className="relative z-10 max-w-4xl mx-auto p-6 sm:p-8">
                <Card className="bg-white/[0.02] backdrop-blur-xl border-white/[0.08]">
                    <CardHeader className="text-center">
                        <div className="mx-auto mb-4 w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/20 to-primary/5 flex items-center justify-center border border-primary/20">
                            <FileText className="w-8 h-8 text-primary" />
                        </div>
                        <CardTitle className="text-3xl font-bold">Terms of Service</CardTitle>
                        <p className="text-muted-foreground mt-2">Last updated: January 2026</p>
                    </CardHeader>

                    <CardContent className="prose prose-invert max-w-none space-y-6">
                        <section>
                            <h2 className="text-xl font-semibold text-foreground mb-4">1. Acceptance of Terms</h2>
                            <p className="text-muted-foreground leading-relaxed">
                                By accessing and using the Chimera Fuzzing Platform, you agree to be bound by these Terms of Service.
                                If you do not agree to these terms, please do not use our services.
                            </p>
                        </section>

                        <section>
                            <h2 className="text-xl font-semibold text-foreground mb-4">2. Use of Services</h2>
                            <p className="text-muted-foreground leading-relaxed">
                                Chimera is a security research platform designed for authorized penetration testing and AI safety research.
                                Users must ensure they have proper authorization before testing any systems.
                            </p>
                            <ul className="list-disc pl-6 mt-3 text-muted-foreground space-y-2">
                                <li>You must be at least 18 years old to use this service</li>
                                <li>You are responsible for maintaining the confidentiality of your account</li>
                                <li>You agree not to use the platform for unauthorized access to any system</li>
                                <li>All research must comply with applicable laws and regulations</li>
                            </ul>
                        </section>

                        <section>
                            <h2 className="text-xl font-semibold text-foreground mb-4">3. Intellectual Property</h2>
                            <p className="text-muted-foreground leading-relaxed">
                                All content, features, and functionality of the Chimera platform are owned by the Chimera team
                                and are protected by international copyright, trademark, and other intellectual property laws.
                            </p>
                        </section>

                        <section>
                            <h2 className="text-xl font-semibold text-foreground mb-4">4. Disclaimer</h2>
                            <p className="text-muted-foreground leading-relaxed">
                                The Chimera platform is provided &quot;as is&quot; without warranties of any kind. We do not guarantee
                                that the service will be uninterrupted, secure, or error-free.
                            </p>
                        </section>

                        <section>
                            <h2 className="text-xl font-semibold text-foreground mb-4">5. Contact</h2>
                            <p className="text-muted-foreground leading-relaxed">
                                For questions about these Terms, please contact us at{" "}
                                <a href="mailto:legal@chimera.dev" className="text-primary hover:underline">
                                    legal@chimera.dev
                                </a>
                            </p>
                        </section>
                    </CardContent>
                </Card>

                {/* Footer */}
                <p className="mt-8 text-center text-sm text-muted-foreground">
                    See also our{" "}
                    <Link href="/privacy" className="text-primary hover:underline">
                        Privacy Policy
                    </Link>
                </p>
            </main>
        </div>
    );
}
