"use client";

import * as React from "react";
import { Grid, List, Search, SlidersHorizontal, Plus } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { 
    TemplateListItem, 
    PromptTemplate,
    SearchTemplatesResponse 
} from "@/types/prompt-library-types";
import { PromptTemplateCard, PromptTemplateCardSkeleton } from "./template-card";

interface PromptTemplateListProps {
    data?: SearchTemplatesResponse;
    isLoading: boolean;
    viewMode?: "grid" | "list";
    onViewModeChange?: (mode: "grid" | "list") => void;
    onTemplateClick?: (template: TemplateListItem) => void;
    onUseTemplate?: (template: TemplateListItem) => void;
    onCreateTemplate?: () => void;
    currentUserId?: string | null;
    className?: string;
}

export function PromptTemplateList({
    data,
    isLoading,
    viewMode = "grid",
    onViewModeChange,
    onTemplateClick,
    onUseTemplate,
    onCreateTemplate,
    currentUserId,
    className,
}: PromptTemplateListProps) {
    if (isLoading) {
        return (
            <div className={cn(
                "grid gap-4",
                viewMode === "grid" 
                    ? "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4" 
                    : "grid-cols-1",
                className
            )}>
                {Array.from({ length: 8 }).map((_, i) => (
                    <PromptTemplateCardSkeleton key={i} variant={viewMode === "grid" ? "default" : "compact"} />
                ))}
            </div>
        );
    }

    if (!data || data.items.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
                <div className="h-12 w-12 rounded-full bg-muted flex items-center justify-center mb-4">
                    <Search className="h-6 w-6 text-muted-foreground" />
                </div>
                <h3 className="text-lg font-semibold">No templates found</h3>
                <p className="text-muted-foreground max-w-xs mt-1">
                    We couldn&apos;t find any prompt templates matching your current filters.
                </p>
                {onCreateTemplate && (
                    <Button onClick={onCreateTemplate} className="mt-6">
                        <Plus className="h-4 w-4 mr-2" />
                        Create First Template
                    </Button>
                )}
            </div>
        );
    }

    return (
        <div className={cn(
            "grid gap-4",
            viewMode === "grid" 
                ? "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4" 
                : "grid-cols-1",
            className
        )}>
            {data.items.map((template) => (
                <PromptTemplateCard
                    key={template.id}
                    template={template}
                    variant={viewMode === "grid" ? "default" : "compact"}
                    currentUserId={currentUserId}
                    onClick={onTemplateClick}
                    onUseTemplate={onUseTemplate}
                />
            ))}
        </div>
    );
}
