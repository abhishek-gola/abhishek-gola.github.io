import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwindcss from '@tailwindcss/vite';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
export default defineConfig({devToolbar:{enabled:false},site: process.env.SITE_URL || 'https://abhishek-gola.github.io', integrations:[mdx(),sitemap()], vite:{plugins:[tailwindcss()]}, markdown:{shikiConfig:{theme:'github-dark'},remarkPlugins:[remarkMath],rehypePlugins:[rehypeKatex]}});
