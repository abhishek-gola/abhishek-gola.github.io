# Abhishek Gola portfolio

A static Astro portfolio for Abhishek Gola’s OpenCV engineering and speaking work.

Production address: https://abhishek-gola.github.io/

## Local development

Use Node 22 and pnpm 11.

```sh
pnpm install --frozen-lockfile
pnpm dev
pnpm build
pnpm verify
```

The build runs Astro diagnostics, generates static pages and writes dist/. Verification checks local links, draft exclusion, canonical metadata, all four recording links, speaking priority and RSS chronology.

## Content

- src/data/speaking.ts: recording metadata, proposed talks and downloadable speaker bio.
- src/data/site.ts: professional links, OpenCV case studies and external writing.
- src/content/writing/: original MDX articles; draft: true excludes an article from published output and feeds.
- public/talks/: official recording thumbnails.
- docs/portfolio-content-checklist.md: included content, optional additions and sources.

Talk topics are proposed session outlines. Previous recordings include team discussions and are not described as solo conference presentations. Recording dates indicate publication month. Case studies preserve public contribution credit and configuration-specific benchmark limitations.

## Deployment

The deploy workflow builds and verifies the source on pushes to main, then publishes dist/ through GitHub Pages. The repository’s Pages source must be GitHub Actions. The site uses its GitHub Pages address as its canonical origin. Override SITE_URL only when moving to a confirmed domain.

The validation workflow checks pull requests without deploying. No database, tracking scripts or paid services are required. Fonts and talk thumbnails are served locally. The contact links open the visitor’s mail app.
