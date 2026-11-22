import rss from '@astrojs/rss';
import type { APIContext } from 'astro';
import { getPublishedWriting } from '../lib/writing';
export async function GET(context: APIContext) {
  return rss({
    title: 'Abhishek Gola — Technical Writing',
    description: 'Computer vision, inference infrastructure and performance engineering.',
    site: import.meta.env.DEV ? context.url.origin : context.site!,
    items: (await getPublishedWriting()).map(entry => ({
      title: entry.title,
      description: `${entry.description} By ${entry.authors}.`,
      pubDate: entry.date,
      link: entry.url,
    })),
  });
}
