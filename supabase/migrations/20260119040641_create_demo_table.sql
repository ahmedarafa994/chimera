-- Create a demo table for testing
CREATE TABLE IF NOT EXISTS public.demo_notes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  content TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE public.demo_notes ENABLE ROW LEVEL SECURITY;

-- Create policy: Users can only see their own notes
CREATE POLICY "Users can view their own notes"
  ON public.demo_notes
  FOR SELECT
  USING (auth.uid() = user_id);

-- Create policy: Users can insert their own notes
CREATE POLICY "Users can insert their own notes"
  ON public.demo_notes
  FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- Create policy: Users can update their own notes
CREATE POLICY "Users can update their own notes"
  ON public.demo_notes
  FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Create policy: Users can delete their own notes
CREATE POLICY "Users can delete their own notes"
  ON public.demo_notes
  FOR DELETE
  USING (auth.uid() = user_id);;
