"use client";

import * as React from "react";
import { 
    PromptTemplate, 
    TemplateVersion, 
    TemplateListItem,
    SharingLevel,
    formatTechniqueType,
    formatVulnerabilityType
} from "@/types/prompt-library-types";
import { 
    Dialog, 
    DialogContent, 
    DialogHeader, 
    DialogTitle,
    DialogDescription,
    DialogFooter
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
    Copy, 
    ExternalLink, 
    Shield, 
    Target, 
    Calendar, 
    User, 
    Lock, 
    Users, 
    Globe,
    Tag,
    History,
    Star,
    Sparkles
} from "lucide-react";
import { TemplateVersionHistory } from "./version-history";
import { TemplateRating } from "./template-rating";
import { usePromptTemplate, useRateTemplate, useCreateVersion } from "@/lib/api/query/prompt-library";
import { useAuth } from "@/hooks/useAuth";
import { toast } from "sonner";

interface TemplateDetailDialogProps {
    templateId: string;
    isOpen: boolean;
    onOpenChange: (open: boolean) => void;
    onUseTemplate?: (template: PromptTemplate) => void;
}

export function TemplateDetailDialog({
    templateId,
    isOpen,
    onOpenChange,
    onUseTemplate,
}: TemplateDetailDialogProps) {
    const { user } = useAuth();
    const { data: template, isLoading } = usePromptTemplate(templateId);
    const rateMutation = useRateTemplate(templateId);
    const versionMutation = useCreateVersion(templateId);

    const handleCopy = async () => {
        if (template) {
            try {
                const currentVersion = template.versions.find(v => v.version_id === template.current_version_id);
                const text = currentVersion?.prompt_text || template.original_prompt;
                await navigator.clipboard.writeText(text);
                toast.success("Prompt copied to clipboard");
            } catch (err) {
                toast.error("Failed to copy prompt");
            }
        }
    };

    const handleRate = (rating: number, effectiveness: boolean, comment?: string) => {
        rateMutation.mutate({ rating, effectiveness_vote: effectiveness, comment }, {
            onSuccess: () => toast.success("Thank you for your rating!"),
            onError: () => toast.error("Failed to submit rating")
        });
    };

    const handleRestore = (version: TemplateVersion) => {
        versionMutation.mutate({ 
            prompt_text: version.prompt_text,
            description: `Restored from version ${version.version_id.substring(0, 8)}`
        }, {
            onSuccess: () => toast.success("Template version restored"),
            onError: () => toast.error("Failed to restore version")
        });
    };

    if (isLoading || !template) {
        return (
            <Dialog open={isOpen} onOpenChange={onOpenChange}>
                <DialogContent className="max-w-3xl h-[80vh] flex items-center justify-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                </DialogContent>
            </Dialog>
        );
    }

    const currentVersion = template.versions.find(v => v.version_id === template.current_version_id);

    return (
        <Dialog open={isOpen} onOpenChange={onOpenChange}>
            <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden flex flex-col p-0 gap-0 border-primary/20 shadow-2xl bg-background/95 backdrop-blur-md">
                <div className="p-6 border-b bg-muted/30">
                    <DialogHeader>
                        <div className="flex items-center gap-3 mb-2">
                            <Badge variant="outline" className="h-6">
                                {template.sharing_level === SharingLevel.PRIVATE ? <Lock className="h-3 w-3 mr-1" /> : 
                                 template.sharing_level === SharingLevel.TEAM ? <Users className="h-3 w-3 mr-1" /> : 
                                 <Globe className="h-3 w-3 mr-1" />}
                                {template.sharing_level.toUpperCase()}
                            </Badge>
                            <Badge className="bg-primary/10 text-primary border-primary/20 hover:bg-primary/20 h-6">
                                <Sparkles className="h-3 w-3 mr-1" /> ACTIVE
                            </Badge>
                        </div>
                        <DialogTitle className="text-2xl font-bold tracking-tight">{template.title}</DialogTitle>
                        <DialogDescription className="text-base mt-1 line-clamp-2">
                            {template.description}
                        </DialogDescription>
                    </DialogHeader>
                </div>

                <div className="flex-1 overflow-y-auto p-6">
                    <Tabs defaultValue="prompt" className="w-full">
                        <div className="flex items-center justify-between mb-6 sticky top-0 bg-background/95 z-10 py-1">
                            <TabsList className="bg-muted/50 p-1">
                                <TabsTrigger value="prompt" className="data-[state=active]:bg-background">
                                    <Sparkles className="h-4 w-4 mr-2" /> Prompt
                                </TabsTrigger>
                                <TabsTrigger value="history" className="data-[state=active]:bg-background">
                                    <History className="h-4 w-4 mr-2" /> History
                                </TabsTrigger>
                                <TabsTrigger value="ratings" className="data-[state=active]:bg-background">
                                    <Star className="h-4 w-4 mr-2" /> Feedback
                                </TabsTrigger>
                            </TabsList>

                            <div className="flex gap-2">
                                <Button variant="outline" size="sm" onClick={handleCopy}>
                                    <Copy className="h-4 w-4 mr-2" /> Copy
                                </Button>
                                <Button size="sm" onClick={() => onUseTemplate?.(template)}>
                                    <ExternalLink className="h-4 w-4 mr-2" /> Use Template
                                </Button>
                            </div>
                        </div>

                        <TabsContent value="prompt" className="space-y-6 mt-0">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div className="space-y-4">
                                    <div>
                                        <h4 className="text-sm font-semibold mb-2 flex items-center gap-2">
                                            <Shield className="h-4 w-4 text-primary" /> Techniques
                                        </h4>
                                        <div className="flex flex-wrap gap-2">
                                            {template.metadata.technique_types.map(t => (
                                                <Badge key={t} variant="secondary">{formatTechniqueType(t)}</Badge>
                                            ))}
                                        </div>
                                    </div>
                                    <div>
                                        <h4 className="text-sm font-semibold mb-2 flex items-center gap-2">
                                            <Target className="h-4 w-4 text-destructive" /> Vulnerabilities
                                        </h4>
                                        <div className="flex flex-wrap gap-2">
                                            {template.metadata.vulnerability_types.map(v => (
                                                <Badge key={v} variant="outline" className="text-destructive border-destructive/30">{formatVulnerabilityType(v)}</Badge>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                                <div className="space-y-4">
                                    <div>
                                        <h4 className="text-sm font-semibold mb-2 flex items-center gap-2">
                                            <Tag className="h-4 w-4 text-blue-500" /> Tags
                                        </h4>
                                        <div className="flex flex-wrap gap-2">
                                            {template.metadata.tags.map(tag => (
                                                <Badge key={tag} variant="outline" className="bg-muted/50">{tag}</Badge>
                                            ))}
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-6 text-sm text-muted-foreground">
                                        <div className="flex items-center gap-1.5">
                                            <User className="h-4 w-4" /> Created by {template.owner_id}
                                        </div>
                                        <div className="flex items-center gap-1.5">
                                            <Calendar className="h-4 w-4" /> {new Date(template.created_at).toLocaleDateString()}
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="space-y-2">
                                <h4 className="text-sm font-semibold">Prompt Text</h4>
                                <div className="p-4 rounded-lg bg-muted/50 border font-mono text-sm leading-relaxed whitespace-pre-wrap">
                                    {currentVersion?.prompt_text || template.original_prompt}
                                </div>
                            </div>
                        </TabsContent>

                        <TabsContent value="history" className="mt-0">
                            <TemplateVersionHistory 
                                versions={template.versions} 
                                currentVersionId={template.current_version_id}
                                onRestore={handleRestore}
                            />
                        </TabsContent>

                        <TabsContent value="ratings" className="mt-0">
                            <TemplateRating 
                                avgRating={template.avg_rating}
                                totalRatings={template.total_ratings}
                                effectivenessScore={template.effectiveness_score}
                                onRate={handleRate}
                                isSubmitting={rateMutation.isPending}
                            />
                        </TabsContent>
                    </Tabs>
                </div>

                <div className="p-4 border-t bg-muted/30 flex justify-end">
                    <Button variant="ghost" onClick={() => onOpenChange(false)}>Close</Button>
                </div>
            </DialogContent>
        </Dialog>
    );
}
