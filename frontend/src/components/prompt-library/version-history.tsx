"use client";

import * as React from "react";
import { Clock, History, ChevronRight, User, Calendar, Copy, RotateCcw } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
    Card,
    CardContent,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import { TemplateVersion } from "@/types/prompt-library-types";
import { Badge } from "@/components/ui/badge";

interface TemplateVersionHistoryProps {
    versions: TemplateVersion[];
    currentVersionId: string;
    onRestore: (version: TemplateVersion) => void;
    className?: string;
}

export function TemplateVersionHistory({
    versions,
    currentVersionId,
    onRestore,
    className,
}: TemplateVersionHistoryProps) {
    const sortedVersions = [...versions].sort((a, b) => 
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    );

    const formatRelativeTime = (dateString: string): string => {
        const date = new Date(dateString);
        const now = new Date();
        const diffMs = now.getTime() - date.getTime();
        const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

        if (diffDays === 0) return "Today";
        if (diffDays === 1) return "Yesterday";
        if (diffDays < 7) return `${diffDays} days ago`;
        return date.toLocaleDateString();
    };

    return (
        <div className={cn("space-y-4", className)}>
            {sortedVersions.map((version, index) => (
                <div key={version.version_id} className="relative pl-6 pb-6 last:pb-0">
                    {/* Timeline line */}
                    {index !== sortedVersions.length - 1 && (
                        <div className="absolute left-2 top-2 bottom-0 w-0.5 bg-muted" />
                    )}
                    
                    {/* Timeline node */}
                    <div className={cn(
                        "absolute left-0 top-1.5 h-4 w-4 rounded-full border-2 border-background",
                        version.version_id === currentVersionId ? "bg-primary" : "bg-muted"
                    )} />

                    <Card className={cn(
                        "transition-all duration-200 hover:border-primary/30",
                        version.version_id === currentVersionId && "border-primary/50 bg-primary/5"
                    )}>
                        <CardHeader className="py-3 px-4 flex flex-row items-center justify-between space-y-0">
                            <div className="flex flex-col gap-1">
                                <div className="flex items-center gap-2">
                                    <span className="font-semibold text-sm">
                                        Version {versions.length - index}
                                    </span>
                                    {version.version_id === currentVersionId && (
                                        <Badge variant="default" className="text-[10px] h-4 px-1.5">Current</Badge>
                                    )}
                                </div>
                                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                                    <span className="flex items-center gap-1">
                                        <Calendar className="h-3 w-3" /> {formatRelativeTime(version.created_at)}
                                    </span>
                                    <span className="flex items-center gap-1">
                                        <User className="h-3 w-3" /> {version.created_by}
                                    </span>
                                </div>
                            </div>
                            {version.version_id !== currentVersionId && (
                                <Button 
                                    variant="ghost" 
                                    size="sm" 
                                    className="h-8"
                                    onClick={() => onRestore(version)}
                                >
                                    <RotateCcw className="h-3.5 w-3.5 mr-1.5" /> Restore
                                </Button>
                            )}
                        </CardHeader>
                        <CardContent className="py-2 px-4">
                            {version.description && (
                                <p className="text-sm text-muted-foreground mb-3 italic">
                                    &quot;{version.description}&quot;
                                </p>
                            )}
                            <div className="bg-muted/50 rounded p-3 font-mono text-xs line-clamp-3">
                                {version.prompt_text}
                            </div>
                        </CardContent>
                    </Card>
                </div>
            ))}
        </div>
    );
}
