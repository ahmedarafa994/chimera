"use client";

import * as React from "react";
import { Star, ThumbsUp, ThumbsDown, MessageSquare } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
    Card,
    CardContent,
    CardHeader,
    CardTitle,
    CardDescription,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

interface TemplateRatingProps {
    avgRating: number;
    totalRatings: number;
    effectivenessScore: number;
    onRate: (rating: number, effectiveness: boolean, comment?: string) => void;
    isSubmitting: boolean;
    userRating?: number;
    userEffectiveness?: boolean;
}

export function TemplateRating({
    avgRating,
    totalRatings,
    effectivenessScore,
    onRate,
    isSubmitting,
    userRating = 0,
    userEffectiveness,
}: TemplateRatingProps) {
    const [rating, setRating] = React.useState(userRating);
    const [effectiveness, setEffectiveness] = React.useState<boolean | undefined>(userEffectiveness);
    const [comment, setComment] = React.useState("");
    const [showForm, setShowForm] = React.useState(false);

    const handleSubmit = () => {
        if (rating > 0 && effectiveness !== undefined) {
            onRate(rating, effectiveness, comment);
            setShowForm(false);
        }
    };

    return (
        <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                            <Star className="h-4 w-4" /> Average Rating
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="flex items-end gap-2">
                            <span className="text-3xl font-bold">{avgRating.toFixed(1)}</span>
                            <span className="text-muted-foreground mb-1">/ 5.0</span>
                        </div>
                        <div className="text-xs text-muted-foreground mt-1">
                            Based on {totalRatings} user reviews
                        </div>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                            <ThumbsUp className="h-4 w-4" /> Effectiveness
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="flex items-end gap-2">
                            <span className="text-3xl font-bold">{(effectivenessScore * 100).toFixed(0)}%</span>
                        </div>
                        <Progress value={effectivenessScore * 100} className="h-2 mt-2" />
                        <div className="text-xs text-muted-foreground mt-1">
                            Percentage of users who found this effective
                        </div>
                    </CardContent>
                </Card>
            </div>

            {!showForm ? (
                <Button variant="outline" className="w-full" onClick={() => setShowForm(true)}>
                    Rate this template
                </Button>
            ) : (
                <Card className="border-primary/50 bg-primary/5">
                    <CardContent className="pt-6 space-y-4">
                        <div className="space-y-2">
                            <label className="text-sm font-medium">Quality Rating</label>
                            <div className="flex gap-1">
                                {[1, 2, 3, 4, 5].map((i) => (
                                    <button
                                        key={i}
                                        type="button"
                                        onClick={() => setRating(i)}
                                        className="p-1 hover:scale-110 transition-transform"
                                    >
                                        <Star 
                                            className={cn(
                                                "h-8 w-8",
                                                i <= rating ? "fill-yellow-400 text-yellow-400" : "text-muted-foreground/30"
                                            )} 
                                        />
                                    </button>
                                ))}
                            </div>
                        </div>

                        <div className="space-y-2">
                            <label className="text-sm font-medium">Was it effective?</label>
                            <div className="flex gap-2">
                                <Button
                                    type="button"
                                    variant={effectiveness === true ? "default" : "outline"}
                                    className="flex-1"
                                    onClick={() => setEffectiveness(true)}
                                >
                                    <ThumbsUp className="h-4 w-4 mr-2" /> Yes
                                </Button>
                                <Button
                                    type="button"
                                    variant={effectiveness === false ? "default" : "outline"}
                                    className="flex-1"
                                    onClick={() => setEffectiveness(false)}
                                >
                                    <ThumbsDown className="h-4 w-4 mr-2" /> No
                                </Button>
                            </div>
                        </div>

                        <div className="space-y-2">
                            <label className="text-sm font-medium">Comments (optional)</label>
                            <Textarea 
                                placeholder="Add your feedback..." 
                                value={comment}
                                onChange={(e) => setComment(e.target.value)}
                            />
                        </div>

                        <div className="flex justify-end gap-2">
                            <Button variant="ghost" onClick={() => setShowForm(false)}>Cancel</Button>
                            <Button onClick={handleSubmit} disabled={isSubmitting || rating === 0 || effectiveness === undefined}>
                                Submit Review
                            </Button>
                        </div>
                    </CardContent>
                </Card>
            )}
        </div>
    );
}
