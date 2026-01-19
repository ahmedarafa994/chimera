"use client";

import * as React from "react";
import { 
    Library, 
    Plus, 
    Grid, 
    List, 
    Sparkles, 
    History,
    Shield,
    BookOpen
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { 
    Card, 
    CardContent, 
    CardHeader, 
    CardTitle, 
    CardDescription 
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PromptTemplateList } from "@/components/prompt-library/template-list";
import { TemplateSearchFilter } from "@/components/prompt-library/search-filter";
import { TemplateDetailDialog } from "@/components/prompt-library/template-detail-dialog";
import { TemplateForm } from "@/components/prompt-library/template-form";
import { 
    Dialog, 
    DialogContent, 
    DialogHeader, 
    DialogTitle,
    DialogDescription 
} from "@/components/ui/dialog";
import { usePromptTemplates, useCreateTemplate } from "@/lib/api/query/prompt-library";
import { SearchTemplatesRequest, TemplateListItem, CreateTemplateRequest } from "@/types/prompt-library-types";
import { useAuth } from "@/hooks/useAuth";
import { toast } from "sonner";

export default function PromptLibraryPage() {
    const { user } = useAuth();
    const [viewMode, setViewMode] = React.useState<"grid" | "list">("grid");
    const [filters, setFilters] = React.useState<SearchTemplatesRequest>({
        limit: 20,
        offset: 0
    });
    
    const [selectedTemplateId, setSelectedTemplateId] = React.useState<string | null>(null);
    const [isDetailOpen, setIsDetailOpen] = React.useState(false);
    const [isCreateOpen, setIsCreateOpen] = React.useState(false);

    const { data, isLoading } = usePromptTemplates(filters);
    const createMutation = useCreateTemplate();

    const handleTemplateClick = (template: TemplateListItem) => {
        setSelectedTemplateId(template.id);
        setIsDetailOpen(true);
    };

    const handleCreateSubmit = (values: any) => {
        createMutation.mutate(values as CreateTemplateRequest, {
            onSuccess: () => {
                toast.success("Template created successfully");
                setIsCreateOpen(false);
            },
            onError: () => toast.error("Failed to create template")
        });
    };

    return (
        <div className="container mx-auto py-8 space-y-8 animate-in fade-in duration-500">
            {/* Header Section */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div className="space-y-1">
                    <div className="flex items-center gap-2">
                        <div className="p-2 bg-primary/10 rounded-lg">
                            <Library className="h-6 w-6 text-primary" />
                        </div>
                        <h1 className="text-3xl font-bold tracking-tight">Prompt Library</h1>
                    </div>
                    <p className="text-muted-foreground text-lg">
                        Centralized repository of verified adversarial prompt templates.
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    <div className="flex bg-muted rounded-lg p-1 border">
                        <Button 
                            variant={viewMode === "grid" ? "secondary" : "ghost"} 
                            size="sm" 
                            className="h-8 w-8 p-0"
                            onClick={() => setViewMode("grid")}
                        >
                            <Grid className="h-4 w-4" />
                        </Button>
                        <Button 
                            variant={viewMode === "list" ? "secondary" : "ghost"} 
                            size="sm" 
                            className="h-8 w-8 p-0"
                            onClick={() => setViewMode("list")}
                        >
                            <List className="h-4 w-4" />
                        </Button>
                    </div>
                    <Button onClick={() => setIsCreateOpen(true)} className="shadow-lg shadow-primary/20">
                        <Plus className="h-4 w-4 mr-2" /> Create Template
                    </Button>
                </div>
            </div>

            {/* Quick Stats / Info Row */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card className="bg-primary/[0.03] border-primary/10">
                    <CardHeader className="py-4">
                        <div className="flex items-center gap-3">
                            <div className="p-2 bg-blue-500/10 rounded-full">
                                <BookOpen className="h-4 w-4 text-blue-500" />
                            </div>
                            <div>
                                <CardTitle className="text-sm font-medium">Total Templates</CardTitle>
                                <CardDescription className="text-2xl font-bold text-foreground">{data?.total || 0}</CardDescription>
                            </div>
                        </div>
                    </CardHeader>
                </Card>
                <Card className="bg-primary/[0.03] border-primary/10">
                    <CardHeader className="py-4">
                        <div className="flex items-center gap-3">
                            <div className="p-2 bg-emerald-500/10 rounded-full">
                                <Shield className="h-4 w-4 text-emerald-500" />
                            </div>
                            <div>
                                <CardTitle className="text-sm font-medium">Verified Effective</CardTitle>
                                <CardDescription className="text-2xl font-bold text-foreground">
                                    {data?.items.filter(i => i.effectiveness_score > 0.7).length || 0}
                                </CardDescription>
                            </div>
                        </div>
                    </CardHeader>
                </Card>
                <Card className="bg-primary/[0.03] border-primary/10">
                    <CardHeader className="py-4">
                        <div className="flex items-center gap-3">
                            <div className="p-2 bg-violet-500/10 rounded-full">
                                <History className="h-4 w-4 text-violet-500" />
                            </div>
                            <div>
                                <CardTitle className="text-sm font-medium">Recent Updates</CardTitle>
                                <CardDescription className="text-2xl font-bold text-foreground">Last 24h</CardDescription>
                            </div>
                        </div>
                    </CardHeader>
                </Card>
            </div>

            {/* Main Content */}
            <div className="space-y-6">
                <TemplateSearchFilter 
                    filters={filters} 
                    onFiltersChange={setFilters} 
                    className="sticky top-4 z-20 bg-background/80 backdrop-blur-sm p-4 rounded-xl border shadow-sm"
                />

                <PromptTemplateList 
                    data={data}
                    isLoading={isLoading}
                    viewMode={viewMode}
                    currentUserId={user?.id}
                    onTemplateClick={handleTemplateClick}
                                        onUseTemplate={(t) => {
                                            setSelectedTemplateId(t.id);
                                            setIsDetailOpen(true);
                                        }}
                    
                    onCreateTemplate={() => setIsCreateOpen(true)}
                />
            </div>

            {/* Dialogs */}
            {selectedTemplateId && (
                <TemplateDetailDialog 
                    templateId={selectedTemplateId}
                    isOpen={isDetailOpen}
                    onOpenChange={setIsDetailOpen}
                />
            )}

            <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
                <DialogContent className="max-w-2xl">
                    <DialogHeader>
                        <DialogTitle>Create Prompt Template</DialogTitle>
                        <DialogDescription>
                            Add a new adversarial prompt to the library.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="py-4">
                        <TemplateForm 
                            onSubmit={handleCreateSubmit}
                            isSubmitting={createMutation.isPending}
                            onCancel={() => setIsCreateOpen(false)}
                        />
                    </div>
                </DialogContent>
            </Dialog>
        </div>
    );
}
