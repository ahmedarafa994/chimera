"use client";

import * as React from "react";
import { Search, X, SlidersHorizontal, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import {
    TechniqueType,
    VulnerabilityType,
    SharingLevel,
    formatTechniqueType,
    formatVulnerabilityType,
    formatSharingLevel,
    SearchTemplatesRequest
} from "@/types/prompt-library-types";

interface TemplateSearchFilterProps {
    filters: SearchTemplatesRequest;
    onFiltersChange: (filters: SearchTemplatesRequest) => void;
    className?: string;
}

export function TemplateSearchFilter({
    filters,
    onFiltersChange,
    className,
}: TemplateSearchFilterProps) {
    const handleQueryChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        onFiltersChange({ ...filters, query: e.target.value, offset: 0 });
    };

    const handleTechniqueChange = (value: string) => {
        onFiltersChange({ 
            ...filters, 
            technique_type: value === "all" ? undefined : value as TechniqueType,
            offset: 0 
        });
    };

    const handleVulnerabilityChange = (value: string) => {
        onFiltersChange({ 
            ...filters, 
            vulnerability_type: value === "all" ? undefined : value as VulnerabilityType,
            offset: 0 
        });
    };

    const handleSharingLevelChange = (value: string) => {
        onFiltersChange({ 
            ...filters, 
            sharing_level: value === "all" ? undefined : value as SharingLevel,
            offset: 0 
        });
    };

    const clearFilters = () => {
        onFiltersChange({
            query: "",
            limit: filters.limit,
            offset: 0
        });
    };

    const hasActiveFilters = !!(filters.query || filters.technique_type || filters.vulnerability_type || filters.sharing_level);

    return (
        <div className={cn("flex flex-col gap-4", className)}>
            <div className="flex flex-col sm:flex-row gap-2">
                <div className="relative flex-1">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                        placeholder="Search templates..."
                        value={filters.query || ""}
                        onChange={handleQueryChange}
                        className="pl-9 pr-4"
                    />
                    {filters.query && (
                        <button
                            onClick={() => onFiltersChange({ ...filters, query: "", offset: 0 })}
                            className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                        >
                            <X className="h-4 w-4" />
                        </button>
                    )}
                </div>
                <div className="flex gap-2">
                    <Select value={filters.technique_type || "all"} onValueChange={handleTechniqueChange}>
                        <SelectTrigger className="w-[160px]">
                            <SelectValue placeholder="Technique" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="all">All Techniques</SelectItem>
                            {Object.values(TechniqueType).map((t) => (
                                <SelectItem key={t} value={t}>{formatTechniqueType(t)}</SelectItem>
                            ))}
                        </SelectContent>
                    </Select>

                    <Select value={filters.vulnerability_type || "all"} onValueChange={handleVulnerabilityChange}>
                        <SelectTrigger className="w-[160px]">
                            <SelectValue placeholder="Vulnerability" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="all">All Vulnerabilities</SelectItem>
                            {Object.values(VulnerabilityType).map((v) => (
                                <SelectItem key={v} value={v}>{formatVulnerabilityType(v)}</SelectItem>
                            ))}
                        </SelectContent>
                    </Select>

                    <Select value={filters.sharing_level || "all"} onValueChange={handleSharingLevelChange}>
                        <SelectTrigger className="w-[140px]">
                            <SelectValue placeholder="Sharing" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="all">All Levels</SelectItem>
                            {Object.values(SharingLevel).map((s) => (
                                <SelectItem key={s} value={s}>{formatSharingLevel(s)}</SelectItem>
                            ))}
                        </SelectContent>
                    </Select>

                    {hasActiveFilters && (
                        <Button variant="ghost" size="icon" onClick={clearFilters} title="Clear all filters">
                            <X className="h-4 w-4" />
                        </Button>
                    )}
                </div>
            </div>
        </div>
    );
}
