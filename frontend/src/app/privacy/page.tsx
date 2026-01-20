"use client";

/**
 * Privacy Policy Page
 *
 * @module app/privacy/page
 */

import Link from "next/link";
import { ArrowLeft, ShieldAlert, Shield } from "lucide-react";
import {
    Card,
    CardContent,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";

export default function PrivacyPage() {
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
                            <Shield className="w-8 h-8 text-primary" />
                        </div>
                        <CardTitle className="text-3xl font-bold">Privacy Policy</CardTitle>
                        <p className="text-muted-foreground mt-2">Last updated: January 2026</p>
                    </CardHeader>

                    <CardContent className="prose prose-invert max-w-none space-y-6">
                        <section>
                            <h2 className="text-xl font-semibold text-foreground mb-4">1. Information We Collect</h2>
                            <p className="text-muted-foreground leading-relaxed">
                                We collect information you provide directly to us, including:
                            </p>
                            <ul className="list-disc pl-6 mt-3 text-muted-foreground space-y-2">
                                <li>Account information (email, username, password)</li>
                                <li>Usage data and session information</li>
                                <li>Research data you choose to store on our platform</li>
                                <li>Communication preferences</li>
                            </ul>
                        </section>

                        <section>
                            <h2 className="text-xl font-semibold text-foreground mb-4">2. How We Use Your Information</h2>
                            <p className="text-muted-foreground leading-relaxed">
                                We use the information we collect to:
                            </p>
                            <ul className="list-disc pl-6 mt-3 text-muted-foreground space-y-2">
                                <li>Provide, maintain, and improve our services</li>
                                <li>Process transactions and send related information</li>
                                <li>Send technical notices and support messages</li>
                                <li>Respond to your comments and questions</li>
                                <li>Detect and prevent fraudulent or unauthorized activity</li>
                            </ul>
                        </section>

                        <section>
                            <h2 className="text-xl font-semibold text-foreground mb-4">3. Data Security</h2>
                            <p className="text-muted-foreground leading-relaxed">
                                We implement industry-standard security measures to protect your personal information.
                                All data is encrypted in transit and at rest. Access to personal data is restricted
                                to authorized personnel only.
                            </p>
                        </section>

                        <section>
                            <h2 className="text-xl font-semibold text-foreground mb-4">4. Data Retention</h2>
                            <p className="text-muted-foreground leading-relaxed">
                                We retain your information for as long as your account is active or as needed to provide
                                services. You may request deletion of your account and associated data at any time.
                            </p>
                        </section>

                        <section>
                            <h2 className="text-xl font-semibold text-foreground mb-4">5. Your Rights</h2>
                            <p className="text-muted-foreground leading-relaxed">
                                You have the right to:
                            </p>
                            <ul className="list-disc pl-6 mt-3 text-muted-foreground space-y-2">
                                <li>Access your personal data</li>
                                <li>Correct inaccurate data</li>
                                <li>Request deletion of your data</li>
                                <li>Export your data in a portable format</li>
                                <li>Opt out of marketing communications</li>
                            </ul>
                        </section>

                        <section>
                            <h2 className="text-xl font-semibold text-foreground mb-4">6. Contact Us</h2>
                            <p className="text-muted-foreground leading-relaxed">
                                For privacy-related inquiries, please contact us at{" "}
                                <a href="mailto:privacy@chimera.dev" className="text-primary hover:underline">
                                    privacy@chimera.dev
                                </a>
                            </p>
                        </section>
                    </CardContent>
                </Card>

                {/* Footer */}
                <p className="mt-8 text-center text-sm text-muted-foreground">
                    See also our{" "}
                    <Link href="/terms" className="text-primary hover:underline">
                        Terms of Service
                    </Link>
                </p>
            </main>
        </div>
    );
}
