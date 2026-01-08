import { useMutation, useQueryClient, UseMutationOptions, QueryKey } from '@tanstack/react-query';
import { toast } from 'sonner';

/**
 * Context type for optimistic mutation
 */
interface OptimisticMutationContext<TData> {
    previousData: TData | undefined;
    [key: string]: unknown;
}

/**
 * Configuration for optimistic mutation
 */
interface OptimisticMutationConfig<
    TData = unknown,
    TError = Error,
    TVariables = void,
    TContext extends OptimisticMutationContext<TData> = OptimisticMutationContext<TData>
> extends Omit<UseMutationOptions<TData, TError, TVariables, TContext>, 'onMutate' | 'onError' | 'onSettled'> {
    queryKey: QueryKey;
    optimisticUpdater: (old: TData | undefined, variables: TVariables) => TData;
    successMessage?: string;
    errorMessage?: string;
    onMutate?: (variables: TVariables) => Promise<Partial<TContext>> | Partial<TContext>;
    onError?: (err: TError, variables: TVariables, context: TContext | undefined) => void;
    onSettled?: (data: TData | undefined, error: TError | null, variables: TVariables, context: TContext | undefined) => void;
}

/**
 * Hook for optimistic mutations with automatic cache updates and rollback
 *
 * @template TData - The data type returned by the mutation
 * @template TError - The error type that may be thrown
 * @template TVariables - The variables type passed to the mutation
 */
export function useOptimisticMutation<
    TData = unknown,
    TError = Error,
    TVariables = void
>(config: OptimisticMutationConfig<TData, TError, TVariables>) {
    const queryClient = useQueryClient();
    const { queryKey, optimisticUpdater, successMessage, errorMessage, ...mutationOptions } = config;

    return useMutation<TData, TError, TVariables, OptimisticMutationContext<TData>>({
        ...mutationOptions,
        onMutate: async (variables: TVariables): Promise<OptimisticMutationContext<TData>> => {
            await queryClient.cancelQueries({ queryKey });
            const previousData = queryClient.getQueryData<TData>(queryKey);
            queryClient.setQueryData<TData>(queryKey, (old) => optimisticUpdater(old, variables));
            if (mutationOptions.onMutate) {
                const ctx = await mutationOptions.onMutate(variables);
                return { ...ctx, previousData };
            }
            return { previousData };
        },
        onError: (err: TError, variables: TVariables, context: OptimisticMutationContext<TData> | undefined) => {
            if (context?.previousData !== undefined) {
                queryClient.setQueryData(queryKey, context.previousData);
            }
            if (errorMessage) toast.error(errorMessage);
            if (mutationOptions.onError) mutationOptions.onError(err, variables, context);
        },
        onSettled: (data: TData | undefined, error: TError | null, variables: TVariables, context: OptimisticMutationContext<TData> | undefined) => {
            queryClient.invalidateQueries({ queryKey });
            if (successMessage && !error) toast.success(successMessage);
            if (mutationOptions.onSettled) mutationOptions.onSettled(data, error, variables, context);
        },
    });
}
