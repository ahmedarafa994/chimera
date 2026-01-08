'use client';

import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";

// Zod validation schema matching SessionConfig interface
const sessionConfigSchema = z.object({
  // AutoDAN Configuration
  populationSize: z.number().int().min(1, "Must be at least 1").max(100, "Cannot exceed 100"),
  numGenerations: z.number().int().min(1, "Must be at least 1").max(1000, "Cannot exceed 1000"),
  mutationRate: z.number().min(0, "Must be between 0 and 1").max(1, "Must be between 0 and 1"),
  crossoverRate: z.number().min(0, "Must be between 0 and 1").max(1, "Must be between 0 and 1"),
  eliteSize: z.number().int().min(0, "Cannot be negative"),
  tournamentSize: z.number().int().min(2, "Must be at least 2"),
  useGradientGuidance: z.boolean(),
  gradientWeight: z.number().min(0, "Must be between 0 and 1").max(1, "Must be between 0 and 1"),

  // Target Configuration
  targetModel: z.string().min(1, "Target model is required"),
  attackObjective: z.string().min(1, "Attack objective is required"),
  initialPrompts: z.array(z.string().min(1, "Prompt cannot be empty")).min(1, "At least one prompt required"),

  // Authorization
  tokenId: z.string().min(1, "Token ID is required"),

  // Optional Settings
  maxIterations: z.number().int().min(1).optional(),
  evaluationFrequency: z.number().int().min(1).optional(),
});

type SessionConfig = z.infer<typeof sessionConfigSchema>;

interface ConfigurationDialogProps {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  onSubmit?: (config: SessionConfig) => void;
  isOpen?: boolean;
  onClose?: () => void;
  className?: string;
}

export function ConfigurationDialog({
  open = false,
  onOpenChange,
  onSubmit,
  isOpen = false,
  onClose,
  className
}: ConfigurationDialogProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [promptInput, setPromptInput] = useState('');

  const dialogOpen = open || isOpen;
  const handleOpenChange = (open: boolean) => {
    if (onOpenChange) onOpenChange(open);
    if (onClose && !open) onClose();
  };

  const form = useForm<SessionConfig>({
    resolver: zodResolver(sessionConfigSchema),
    defaultValues: {
      populationSize: 20,
      numGenerations: 10,
      mutationRate: 0.3,
      crossoverRate: 0.7,
      eliteSize: 5,
      tournamentSize: 3,
      useGradientGuidance: false,
      gradientWeight: 0.5,
      targetModel: 'gpt-4',
      attackObjective: '',
      initialPrompts: [],
      tokenId: '',
    },
  });

  const { register, handleSubmit: handleFormSubmit, formState: { errors }, watch, setValue } = form;
  const useGradientGuidance = watch('useGradientGuidance');
  const initialPrompts = watch('initialPrompts');

  const handleSubmit = (data: SessionConfig) => {
    if (onSubmit) onSubmit(data);
    handleOpenChange(false);
  };

  const addPrompt = () => {
    if (promptInput.trim()) {
      setValue('initialPrompts', [...initialPrompts, promptInput.trim()]);
      setPromptInput('');
    }
  };

  const removePrompt = (index: number) => {
    setValue('initialPrompts', initialPrompts.filter((_, i) => i !== index));
  };

  return (
    <Dialog open={dialogOpen} onOpenChange={handleOpenChange}>
      <DialogContent className={`max-w-2xl max-h-[90vh] overflow-y-auto ${className}`}>
        <DialogHeader>
          <DialogTitle>Configure DeepTeam Session</DialogTitle>
          <DialogDescription>
            Set up your multi-agent collaborative red-teaming session with AutoDAN optimization
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleFormSubmit(handleSubmit)} className="space-y-6">
          {/* Basic Configuration */}
          <div className="space-y-4">
            <h3 className="text-sm font-semibold">Target Configuration</h3>

            <div className="space-y-2">
              <Label htmlFor="targetModel">Target Model</Label>
              <Input
                id="targetModel"
                placeholder="e.g., gpt-4, claude-3-opus"
                {...register('targetModel')}
                aria-invalid={!!errors.targetModel}
              />
              {errors.targetModel && (
                <p className="text-xs text-destructive">{errors.targetModel.message}</p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="attackObjective">Attack Objective</Label>
              <Input
                id="attackObjective"
                placeholder="Describe the red-teaming objective"
                {...register('attackObjective')}
                aria-invalid={!!errors.attackObjective}
              />
              {errors.attackObjective && (
                <p className="text-xs text-destructive">{errors.attackObjective.message}</p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="tokenId">Authorization Token ID</Label>
              <Input
                id="tokenId"
                type="password"
                placeholder="Enter your authorization token"
                {...register('tokenId')}
                aria-invalid={!!errors.tokenId}
              />
              {errors.tokenId && (
                <p className="text-xs text-destructive">{errors.tokenId.message}</p>
              )}
            </div>

            <div className="space-y-2">
              <Label>Initial Prompts</Label>
              <div className="flex gap-2">
                <Input
                  placeholder="Enter a prompt and click Add"
                  value={promptInput}
                  onChange={(e) => setPromptInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), addPrompt())}
                />
                <Button type="button" onClick={addPrompt} variant="outline">Add</Button>
              </div>
              <div className="space-y-1">
                {initialPrompts.map((prompt, index) => (
                  <div key={index} className="flex items-center gap-2 text-sm p-2 bg-muted rounded">
                    <span className="flex-1 truncate">{prompt}</span>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => removePrompt(index)}
                    >
                      Remove
                    </Button>
                  </div>
                ))}
              </div>
              {errors.initialPrompts && (
                <p className="text-xs text-destructive">{errors.initialPrompts.message}</p>
              )}
            </div>
          </div>

          {/* AutoDAN Parameters */}
          <div className="space-y-4">
            <h3 className="text-sm font-semibold">AutoDAN Genetic Algorithm</h3>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="populationSize">Population Size</Label>
                <Input
                  id="populationSize"
                  type="number"
                  {...register('populationSize', { valueAsNumber: true })}
                  aria-invalid={!!errors.populationSize}
                />
                {errors.populationSize && (
                  <p className="text-xs text-destructive">{errors.populationSize.message}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="numGenerations">Generations</Label>
                <Input
                  id="numGenerations"
                  type="number"
                  {...register('numGenerations', { valueAsNumber: true })}
                  aria-invalid={!!errors.numGenerations}
                />
                {errors.numGenerations && (
                  <p className="text-xs text-destructive">{errors.numGenerations.message}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="eliteSize">Elite Size</Label>
                <Input
                  id="eliteSize"
                  type="number"
                  {...register('eliteSize', { valueAsNumber: true })}
                  aria-invalid={!!errors.eliteSize}
                />
                {errors.eliteSize && (
                  <p className="text-xs text-destructive">{errors.eliteSize.message}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="tournamentSize">Tournament Size</Label>
                <Input
                  id="tournamentSize"
                  type="number"
                  {...register('tournamentSize', { valueAsNumber: true })}
                  aria-invalid={!!errors.tournamentSize}
                />
                {errors.tournamentSize && (
                  <p className="text-xs text-destructive">{errors.tournamentSize.message}</p>
                )}
              </div>
            </div>

            <div className="space-y-3">
              <div className="space-y-2">
                <Label htmlFor="mutationRate">Mutation Rate: {watch('mutationRate').toFixed(2)}</Label>
                <Slider
                  id="mutationRate"
                  min={0}
                  max={1}
                  step={0.05}
                  value={[watch('mutationRate')]}
                  onValueChange={(value) => setValue('mutationRate', value[0])}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="crossoverRate">Crossover Rate: {watch('crossoverRate').toFixed(2)}</Label>
                <Slider
                  id="crossoverRate"
                  min={0}
                  max={1}
                  step={0.05}
                  value={[watch('crossoverRate')]}
                  onValueChange={(value) => setValue('crossoverRate', value[0])}
                />
              </div>
            </div>
          </div>

          {/* Gradient Guidance */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label htmlFor="useGradientGuidance">Gradient Guidance</Label>
                <p className="text-xs text-muted-foreground">Enable gradient-based optimization</p>
              </div>
              <Switch
                id="useGradientGuidance"
                checked={useGradientGuidance}
                onCheckedChange={(checked) => setValue('useGradientGuidance', checked)}
              />
            </div>

            {useGradientGuidance && (
              <div className="space-y-2 pl-4 border-l-2 border-muted">
                <Label htmlFor="gradientWeight">Gradient Weight: {watch('gradientWeight').toFixed(2)}</Label>
                <Slider
                  id="gradientWeight"
                  min={0}
                  max={1}
                  step={0.05}
                  value={[watch('gradientWeight')]}
                  onValueChange={(value) => setValue('gradientWeight', value[0])}
                />
              </div>
            )}
          </div>

          {/* Advanced Settings Toggle */}
          <div className="flex items-center justify-between pt-2 border-t">
            <Button
              type="button"
              variant="ghost"
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
            </Button>
          </div>

          {/* Advanced Settings */}
          {showAdvanced && (
            <div className="space-y-4 p-4 bg-muted/50 rounded-lg">
              <h3 className="text-sm font-semibold">Advanced Settings</h3>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="maxIterations">Max Iterations (Optional)</Label>
                  <Input
                    id="maxIterations"
                    type="number"
                    {...register('maxIterations', { valueAsNumber: true })}
                    aria-invalid={!!errors.maxIterations}
                  />
                  {errors.maxIterations && (
                    <p className="text-xs text-destructive">{errors.maxIterations.message}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="evaluationFrequency">Evaluation Frequency (Optional)</Label>
                  <Input
                    id="evaluationFrequency"
                    type="number"
                    {...register('evaluationFrequency', { valueAsNumber: true })}
                    aria-invalid={!!errors.evaluationFrequency}
                  />
                  {errors.evaluationFrequency && (
                    <p className="text-xs text-destructive">{errors.evaluationFrequency.message}</p>
                  )}
                </div>
              </div>
            </div>
          )}

          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => handleOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit">Start Session</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

export default ConfigurationDialog;
