"use client";

import * as React from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { 
    TechniqueType, 
    VulnerabilityType, 
    SharingLevel,
    CreateTemplateRequest,
    UpdateTemplateRequest,
    PromptTemplate
} from "@/types/prompt-library-types";
import { Button } from "@/components/ui/button";
import {
    Form,
    FormControl,
    FormDescription,
    FormField,
    FormItem,
    FormLabel,
    FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { X, Plus, Sparkles } from "lucide-react";

const formSchema = z.object({
    title: z.string().min(3, "Title must be at least 3 characters"),
    description: z.string().min(10, "Description must be at least 10 characters"),
    prompt_text: z.string().min(10, "Prompt must be at least 10 characters"),
    technique_types: z.array(z.nativeEnum(TechniqueType)).min(1, "Select at least one technique"),
    vulnerability_types: z.array(z.nativeEnum(VulnerabilityType)).min(1, "Select at least one vulnerability"),
    sharing_level: z.nativeEnum(SharingLevel),
    tags: z.array(z.string()),
});

interface TemplateFormProps {
    initialData?: PromptTemplate;
    onSubmit: (data: CreateTemplateRequest | UpdateTemplateRequest) => void;
    isSubmitting: boolean;
    onCancel?: () => void;
}

export function TemplateForm({
    initialData,
    onSubmit,
    isSubmitting,
    onCancel,
}: TemplateFormProps) {
    const [tagInput, setTagInput] = React.useState("");

    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            title: initialData?.title || "",
            description: initialData?.description || "",
            prompt_text: initialData?.original_prompt || "",
            technique_types: initialData?.metadata.technique_types || [],
            vulnerability_types: initialData?.metadata.vulnerability_types || [],
            sharing_level: initialData?.sharing_level || SharingLevel.PRIVATE,
            tags: initialData?.metadata.tags || [],
        },
    });

    const handleSubmit = (values: z.infer<typeof formSchema>) => {
        onSubmit(values as CreateTemplateRequest);
    };

    const addTag = () => {
        if (tagInput.trim()) {
            const currentTags = form.getValues("tags");
            if (!currentTags.includes(tagInput.trim())) {
                form.setValue("tags", [...currentTags, tagInput.trim()]);
            }
            setTagInput("");
        }
    };

    const removeTag = (tag: string) => {
        const currentTags = form.getValues("tags");
        form.setValue("tags", currentTags.filter((t) => t !== tag));
    };

    return (
        <Form {...form}>
            <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-6">
                <FormField
                    control={form.control}
                    name="title"
                    render={({ field }) => (
                        <FormItem>
                            <FormLabel>Title</FormLabel>
                            <FormControl>
                                <Input placeholder="e.g., Quantum Bypass Alpha" {...field} />
                            </FormControl>
                            <FormDescription>
                                Give your template a clear, unique name.
                            </FormDescription>
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
                                    placeholder="Explain what this prompt does and why it works..." 
                                    className="min-h-[100px]"
                                    {...field} 
                                />
                            </FormControl>
                            <FormMessage />
                        </FormItem>
                    )}
                />

                <FormField
                    control={form.control}
                    name="prompt_text"
                    render={({ field }) => (
                        <FormItem>
                            <FormLabel>Prompt Template</FormLabel>
                            <FormControl>
                                <Textarea 
                                    placeholder="Enter the template text. Use placeholders if needed." 
                                    className="min-h-[200px] font-mono text-sm"
                                    {...field} 
                                />
                            </FormControl>
                            <FormDescription>
                                This is the core adversarial prompt.
                            </FormDescription>
                            <FormMessage />
                        </FormItem>
                    )}
                />

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <FormField
                        control={form.control}
                        name="sharing_level"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Sharing Level</FormLabel>
                                <Select onValueChange={field.onChange} defaultValue={field.value}>
                                    <FormControl>
                                        <SelectTrigger>
                                            <SelectValue placeholder="Select visibility" />
                                        </SelectTrigger>
                                    </FormControl>
                                    <SelectContent>
                                        {Object.values(SharingLevel).map((s) => (
                                            <SelectItem key={s} value={s}>
                                                {s.charAt(0).toUpperCase() + s.slice(1)}
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                                <FormMessage />
                            </FormItem>
                        )}
                    />

                    <FormItem>
                        <FormLabel>Tags</FormLabel>
                        <div className="flex gap-2">
                            <Input 
                                value={tagInput}
                                onChange={(e) => setTagInput(e.target.value)}
                                placeholder="Add a tag..."
                                onKeyDown={(e) => {
                                    if (e.key === "Enter") {
                                        e.preventDefault();
                                        addTag();
                                    }
                                }}
                            />
                            <Button type="button" variant="outline" size="icon" onClick={addTag}>
                                <Plus className="h-4 w-4" />
                            </Button>
                        </div>
                        <div className="flex flex-wrap gap-2 mt-2">
                            {form.watch("tags").map((tag) => (
                                <Badge key={tag} variant="secondary" className="pl-2 pr-1">
                                    {tag}
                                    <button 
                                        type="button" 
                                        onClick={() => removeTag(tag)}
                                        className="ml-1 hover:bg-muted rounded-full p-0.5"
                                    >
                                        <X className="h-3 w-3" />
                                    </button>
                                </Badge>
                            ))}
                        </div>
                    </FormItem>
                </div>

                <div className="flex justify-end gap-2 pt-4">
                    {onCancel && (
                        <Button type="button" variant="ghost" onClick={onCancel}>
                            Cancel
                        </Button>
                    )}
                    <Button type="submit" disabled={isSubmitting}>
                        {isSubmitting ? "Saving..." : (initialData ? "Update Template" : "Create Template")}
                    </Button>
                </div>
            </form>
        </Form>
    );
}
