import { useQuery, useMutation, useQueryClient, useInfiniteQuery } from "@tanstack/react-query";
import axios from "axios";
import {
    SearchTemplatesRequest,
    SearchTemplatesResponse,
    PromptTemplate,
    CreateTemplateRequest,
    UpdateTemplateRequest,
    RateTemplateRequest,
    CreateVersionRequest,
    TemplateVersion,
    TemplateRating,
    SaveFromCampaignRequest
} from "@/types/prompt-library-types";

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || '/api/v1') + "/prompt-library/templates";

export const promptLibraryKeys = {
    all: ["prompt-library"] as const,
    lists: () => [...promptLibraryKeys.all, "list"] as const,
    list: (filters: SearchTemplatesRequest) => [...promptLibraryKeys.lists(), filters] as const,
    details: () => [...promptLibraryKeys.all, "detail"] as const,
    detail: (id: string) => [...promptLibraryKeys.details(), id] as const,
    versions: (id: string) => [...promptLibraryKeys.detail(id), "versions"] as const,
    ratings: (id: string) => [...promptLibraryKeys.detail(id), "ratings"] as const,
    statistics: () => [...promptLibraryKeys.all, "statistics"] as const,
};

// Query hooks
export const usePromptTemplates = (filters: SearchTemplatesRequest) => {
    return useQuery({
        queryKey: promptLibraryKeys.list(filters),
        queryFn: async () => {
            const { data } = await axios.post<SearchTemplatesResponse>(`${API_BASE}/search`, filters);
            return data;
        },
    });
};

export const usePromptTemplate = (id: string) => {
    return useQuery({
        queryKey: promptLibraryKeys.detail(id),
        queryFn: async () => {
            const { data } = await axios.get<PromptTemplate>(`${API_BASE}/${id}`);
            return data;
        },
        enabled: !!id,
    });
};

export const useSearchTemplates = (filters: SearchTemplatesRequest) => {
    return useQuery({
        queryKey: promptLibraryKeys.list(filters),
        queryFn: async () => {
            const { data } = await axios.post<SearchTemplatesResponse>(`${API_BASE}/search`, filters);
            return data;
        },
    });
};

export const useInfiniteSearchTemplates = (filters: Omit<SearchTemplatesRequest, "offset">) => {
    return useInfiniteQuery({
        queryKey: [...promptLibraryKeys.lists(), "infinite", filters],
        queryFn: async ({ pageParam = 0 }) => {
            const { data } = await axios.post<SearchTemplatesResponse>(`${API_BASE}/search`, {
                ...filters,
                offset: pageParam,
            });
            return data;
        },
        initialPageParam: 0,
        getNextPageParam: (lastPage) => {
            const nextOffset = lastPage.offset + lastPage.limit;
            return nextOffset < lastPage.total ? nextOffset : undefined;
        },
    });
};

export const useTopRatedTemplates = (limit: number = 5) => {
    return useQuery({
        queryKey: [...promptLibraryKeys.lists(), "top-rated", limit],
        queryFn: async () => {
            const { data } = await axios.get<SearchTemplatesResponse>(`${API_BASE}/top-rated?limit=${limit}`);
            return data;
        },
    });
};

export const useLibraryStatistics = () => {
    return useQuery({
        queryKey: promptLibraryKeys.statistics(),
        queryFn: async () => {
            const { data } = await axios.get(`${API_BASE}/statistics`);
            return data;
        },
    });
};

export const useTemplateVersions = (id: string) => {
    return useQuery({
        queryKey: promptLibraryKeys.versions(id),
        queryFn: async () => {
            const { data } = await axios.get<TemplateVersion[]>(`${API_BASE}/${id}/versions`);
            return data;
        },
        enabled: !!id,
    });
};

export const useTemplateVersion = (templateId: string, versionId: string) => {
    return useQuery({
        queryKey: [...promptLibraryKeys.versions(templateId), versionId],
        queryFn: async () => {
            const { data } = await axios.get<TemplateVersion>(`${API_BASE}/${templateId}/versions/${versionId}`);
            return data;
        },
        enabled: !!templateId && !!versionId,
    });
};

export const useTemplateRatings = (id: string) => {
    return useQuery({
        queryKey: promptLibraryKeys.ratings(id),
        queryFn: async () => {
            const { data } = await axios.get<TemplateRating[]>(`${API_BASE}/${id}/ratings`);
            return data;
        },
        enabled: !!id,
    });
};

export const useRatingStatistics = (id: string) => {
    return useQuery({
        queryKey: [...promptLibraryKeys.ratings(id), "stats"],
        queryFn: async () => {
            const { data } = await axios.get(`${API_BASE}/${id}/ratings/stats`);
            return data;
        },
        enabled: !!id,
    });
};

// Mutation hooks
export const useCreateTemplate = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async (request: CreateTemplateRequest) => {
            const { data } = await axios.post<PromptTemplate>(`${API_BASE}`, request);
            return data;
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: promptLibraryKeys.lists() });
        },
    });
};

export const useUpdateTemplate = (id: string) => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async (request: UpdateTemplateRequest) => {
            const { data } = await axios.put<PromptTemplate>(`${API_BASE}/${id}`, request);
            return data;
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: promptLibraryKeys.detail(id) });
            queryClient.invalidateQueries({ queryKey: promptLibraryKeys.lists() });
        },
    });
};

export const useDeleteTemplate = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async (id: string) => {
            await axios.delete(`${API_BASE}/${id}`);
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: promptLibraryKeys.lists() });
        },
    });
};

export const useRateTemplate = (id: string) => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async (request: RateTemplateRequest) => {
            const { data } = await axios.post<PromptTemplate>(`${API_BASE}/${id}/rate`, request);
            return data;
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: promptLibraryKeys.detail(id) });
            queryClient.invalidateQueries({ queryKey: promptLibraryKeys.ratings(id) });
        },
    });
};

export const useUpdateRating = (templateId: string) => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async (request: RateTemplateRequest) => {
            const { data } = await axios.put(`${API_BASE}/${templateId}/rate`, request);
            return data;
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: promptLibraryKeys.detail(templateId) });
            queryClient.invalidateQueries({ queryKey: promptLibraryKeys.ratings(templateId) });
        },
    });
};

export const useDeleteRating = (templateId: string) => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async () => {
            await axios.delete(`${API_BASE}/${templateId}/rate`);
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: promptLibraryKeys.detail(templateId) });
            queryClient.invalidateQueries({ queryKey: promptLibraryKeys.ratings(templateId) });
        },
    });
};

export const useCreateVersion = (id: string) => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async (request: CreateVersionRequest) => {
            const { data } = await axios.post<PromptTemplate>(`${API_BASE}/${id}/versions`, request);
            return data;
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: promptLibraryKeys.detail(id) });
            queryClient.invalidateQueries({ queryKey: promptLibraryKeys.versions(id) });
        },
    });
};

export const useRestoreVersion = (templateId: string) => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async (versionId: string) => {
            const { data } = await axios.post(`${API_BASE}/${templateId}/versions/${versionId}/restore`);
            return data;
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: promptLibraryKeys.detail(templateId) });
            queryClient.invalidateQueries({ queryKey: promptLibraryKeys.versions(templateId) });
        },
    });
};

export const useSaveFromCampaign = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async (request: SaveFromCampaignRequest) => {
            const { data } = await axios.post<PromptTemplate>(`${API_BASE}/save-from-campaign`, request);
            return data;
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: promptLibraryKeys.lists() });
        },
    });
};

// Utility hooks
export const usePrefetchTemplate = (id: string) => {
    const queryClient = useQueryClient();
    return () => queryClient.prefetchQuery({
        queryKey: promptLibraryKeys.detail(id),
        queryFn: async () => {
            const { data } = await axios.get<PromptTemplate>(`${API_BASE}/${id}`);
            return data;
        },
    });
};

export const usePrefetchTemplateVersions = (id: string) => {
    const queryClient = useQueryClient();
    return () => queryClient.prefetchQuery({
        queryKey: promptLibraryKeys.versions(id),
        queryFn: async () => {
            const { data } = await axios.get<TemplateVersion[]>(`${API_BASE}/${id}/versions`);
            return data;
        },
    });
};

export const useInvalidatePromptLibrary = () => {
    const queryClient = useQueryClient();
    return () => queryClient.invalidateQueries({ queryKey: promptLibraryKeys.all });
};
