"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";
import { AlertCircle, RefreshCcw, Home } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";

interface Props {
    children?: ReactNode;
    fallback?: ReactNode;
}

interface State {
    hasError: boolean;
    error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
    public state: State = {
        hasError: false,
        error: null,
    };

    public static getDerivedStateFromError(error: Error): State {
        // Update state so the next render will show the fallback UI.
        return { hasError: true, error };
    }

    public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error("Uncaught error:", error, errorInfo);
        // You could also log the error to an error reporting service here
    }

    private handleReset = () => {
        this.setState({ hasError: false, error: null });
        window.location.reload();
    };

    private handleGoHome = () => {
        this.setState({ hasError: false, error: null });
        window.location.href = "/";
    };

    public render() {
        if (this.state.hasError) {
            if (this.props.fallback) {
                return this.props.fallback;
            }

            return (
                <div className="min-h-screen flex items-center justify-center p-4 bg-slate-950">
                    <Card className="max-w-md w-full border-red-500/20 bg-slate-900/50 backdrop-blur-xl">
                        <CardHeader className="text-center">
                            <div className="mx-auto mb-4 h-12 w-12 rounded-full bg-red-500/10 flex items-center justify-center">
                                <AlertCircle className="h-6 w-6 text-red-500" />
                            </div>
                            <CardTitle className="text-xl font-bold text-slate-100">Something went wrong</CardTitle>
                            <CardDescription className="text-slate-400 mt-2">
                                An unexpected error occurred while rendering this page.
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="p-3 rounded bg-slate-800/50 border border-slate-700 text-xs font-mono text-red-400 overflow-auto max-h-32">
                                {this.state.error?.name}: {this.state.error?.message}
                            </div>
                        </CardContent>
                        <CardFooter className="flex flex-col gap-2">
                            <Button
                                onClick={this.handleReset}
                                className="w-full bg-slate-200 text-slate-900 hover:bg-slate-300 transition-colors"
                                variant="default"
                            >
                                <RefreshCcw className="mr-2 h-4 w-4" />
                                Try Again
                            </Button>
                            <Button
                                onClick={this.handleGoHome}
                                variant="ghost"
                                className="w-full text-slate-400 hover:text-slate-200 hover:bg-slate-800"
                            >
                                <Home className="mr-2 h-4 w-4" />
                                Return to Dashboard
                            </Button>
                        </CardFooter>
                    </Card>
                </div>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
