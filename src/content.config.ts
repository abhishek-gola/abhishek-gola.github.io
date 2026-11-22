import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const writing = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/writing' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    topic: z.string(),
    draft: z.boolean().default(false),
    authors: z.array(z.string()).min(1).default(['Abhishek Gola']),
    related: z.array(z.string()).default([]),
  }),
});
export const collections = { writing };
