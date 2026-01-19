"use client";

import * as React from "react";
import { 
    Dialog, 
    DialogContent, 
    DialogHeader, 
    DialogTitle,
    DialogDescription,
    DialogFooter
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { 
    Form,
    FormControl,
    FormField,
    FormItem,
    FormLabel,
    FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { 
    TechniqueType, 
    VulnerabilityType, 
    SharingLevel,
    CreateTemplateRequest 
} from "@/types/prompt-library-types";
import { useCreateTemplate } from "@/lib/api/query/prompt-library";
import { toast } from "sonner";
import { Library, Sparkles } from "lucide-react";

const formSchema = z.object({
    title: z.string().min(3, "Title must be at least 3 characters"),
    description: z.string().min(10, "Description must be at least 10 characters"),
    sharing_level: z.nativeEnum(SharingLevel),
});

interface SaveToLibraryDialogProps {
    promptText: string;
    techniqueTypes: TechniqueType[];
    vulnerabilityTypes: VulnerabilityType[];
    targetModels: string[];
    tags: string[];
    isOpen: boolean;
    onOpenChange: (open: boolean) => void;
}

export function SaveToLibraryDialog({
    promptText,
    techniqueTypes,
    vulnerabilityTypes,
    targetModels,
    tags,
    isOpen,
    onOpenChange,
}: SaveToLibraryDialogProps) {
    const createMutation = useCreateTemplate();

    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            title: "",
            description: "",
            sharing_level: SharingLevel.PRIVATE,
        },
    });

    const onSubmit = (values: z.infer<typeof formSchema>) => {
        const request: CreateTemplateRequest = {
            ...values,
            prompt_text: promptText,
            technique_types: techniqueTypes,
            vulnerability_types: vulnerabilityTypes,
            target_models: targetModels,
            tags: tags,
        };

        createMutation.mutate(request, {
            onSuccess: () => {
                toast.success("Prompt saved to library successfully");
                onOpenChange(false);
                form.reset();
            },
            onError: (err) => {
                toast.error("Failed to save template to library");
            }
        });
    };

    return (
        <Dialog open={isOpen} onOpenChange={onOpenChange}>
            <DialogContent className="max-w-md">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <Library className="h-5 w-5 text-primary" /> Save to Prompt Library
                    </DialogTitle>
                    <DialogDescription>
                        Convert this successful prompt into a reusable template.
                    </DialogDescription>
                </DialogHeader>

                <Form {...form}>
                    <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4 py-2">
                        <FormField
                            control={form.control}
                            name="title"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel>Template Title</FormLabel>
                                    <FormControl>
                                        <Input placeholder="e.g., GPT-4 Jailbreak - Method A" {...field} />
                                    </FormControl>
                                    <FormMessage />
                                </FormItem>
                            )}
                        />

                        <FormField
                            control={form.control}
                            name="description"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel>Description</FormLabel>
                                    <FormControl>
                                        <Textarea 
                                            placeholder="What makes this prompt effective?" 
                                            className="min-h-[80px]"
                                            {...field} 
                                        />
                                    </FormControl>
                                    <FormMessage />
                                </FormItem>
                            )}
                        />

                        <div className="p-3 rounded bg-muted/50 font-mono text-[10px] max-h-[100px] overflow-hidden opacity-70 border">
                            {promptText}
                        </div>

                        <DialogFooter className="pt-4">
                            <Button type="button" variant="ghost" onClick={() => onOpenChange(false)}>
                                Cancel
                            </Button>
                            <Button type="submit" disabled={createMutation.isPending}>
                                {createMutation.isPending ? "Saving..." : "Save Template"}
                            </Button>
                        </DialogFooter>
                    </form>
                </Form>
            </DialogContent>
        </Dialog>
    );
}
