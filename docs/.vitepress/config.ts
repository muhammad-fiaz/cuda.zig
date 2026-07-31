import { defineConfig } from "vitepress";
import llmstxt from "vitepress-plugin-llms";

export const SITE_URL = "https://muhammad-fiaz.github.io/cuda.zig";
export const SITE_NAME = "cuda.zig";
export const SITE_DESCRIPTION =
  "GPU Computing and CUDA Runtime API library for Zig. Zero link-time dependencies, automatic CPU fallback, CUDA 12.x and 13.x support, tensor operations, NVRTC compilation, and multi-GPU peer access.";

export const GA_ID = "G-6BVYCRK57P";
export const GTM_ID = "GTM-P4M9T8ZR";
export const ADSENSE_CLIENT_ID = "ca-pub-2040560600290490";

export const KEYWORDS =
  "zig, cuda, gpu, nvidia, nvrtc, device memory, streams, events, tensors, matmul, fallback, parallel computing, cuda runtime, driver api";

export default defineConfig({
  lang: "en-US",
  title: SITE_NAME,
  titleTemplate: `:title | ${SITE_NAME}`,
  description: SITE_DESCRIPTION,
  base: "/cuda.zig/",
  lastUpdated: true,
  cleanUrls: false,

  sitemap: {
    hostname: SITE_URL,
  },

  vite: {
    plugins: [llmstxt()],
  },

  head: [
    ["meta", { name: "title", content: SITE_NAME }],
    ["meta", { name: "description", content: SITE_DESCRIPTION }],
    ["meta", { name: "keywords", content: KEYWORDS }],
    ["meta", { name: "author", content: "Muhammad Fiaz" }],
    ["meta", { name: "robots", content: "index, follow" }],
    ["meta", { name: "language", content: "English" }],
    ["meta", { name: "revisit-after", content: "7 days" }],

    // Open Graph
    ["meta", { property: "og:type", content: "website" }],
    ["meta", { property: "og:url", content: SITE_URL }],
    ["meta", { property: "og:title", content: "GPU Computing for Zig | cuda.zig" }],
    ["meta", { property: "og:description", content: SITE_DESCRIPTION }],
    ["meta", { property: "og:image", content: `${SITE_URL}/cover.png` }],
    ["meta", { property: "og:image:width", content: "1200" }],
    ["meta", { property: "og:image:height", content: "630" }],
    ["meta", { property: "og:image:alt", content: "cuda.zig — GPU Computing for Zig" }],
    ["meta", { property: "og:site_name", content: SITE_NAME }],
    ["meta", { property: "og:locale", content: "en_US" }],

    // Twitter Card
    ["meta", { name: "twitter:card", content: "summary_large_image" }],
    ["meta", { name: "twitter:url", content: SITE_URL }],
    ["meta", { name: "twitter:title", content: "GPU Computing for Zig | cuda.zig" }],
    ["meta", { name: "twitter:description", content: SITE_DESCRIPTION }],
    ["meta", { name: "twitter:image", content: `${SITE_URL}/cover.png` }],
    ["meta", { name: "twitter:image:alt", content: "cuda.zig — GPU Computing for Zig" }],
    ["meta", { name: "twitter:site", content: "@muhammadfiaz_" }],
    ["meta", { name: "twitter:creator", content: "@muhammadfiaz_" }],

    // Canonical
    ["link", { rel: "canonical", href: SITE_URL }],

    // Favicon
    ["link", { rel: "icon", href: "/cuda.zig/favicon.png", type: "image/png" }],
    ["link", { rel: "apple-touch-icon", href: "/cuda.zig/favicon.png" }],
    ["link", { rel: "manifest", href: "/cuda.zig/site.webmanifest" }],

    // Theme
    ["meta", { name: "theme-color", content: "#76b900" }],
    ["meta", { name: "msapplication-TileColor", content: "#76b900" }],

    // Google Analytics
    ["script", { async: "", src: `https://www.googletagmanager.com/gtag/js?id=${GA_ID}` }],
    [
      "script",
      {},
      `window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag('js',new Date());gtag('config','${GA_ID}');`,
    ],

    // Google Tag Manager
    [
      "script",
      {},
      `(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src='https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);})(window,document,'script','dataLayer','${GTM_ID}');`,
    ],

    // AdSense
    [
      "script",
      {
        async: "",
        src: `https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=${ADSENSE_CLIENT_ID}`,
        crossorigin: "anonymous",
      },
    ],
  ],

  ignoreDeadLinks: [/.*\.zig$/],

  transformPageData(pageData: any) {
    const pageTitle = pageData.title || SITE_NAME;
    const pageDescription = pageData.description || SITE_DESCRIPTION;
    const normalizedPath = pageData.relativePath
      .replace(/\.md$/, "")
      .replace(/(^|\/)index$/, "$1")
      .replace(/\/$/, "");
    const canonicalUrl =
      normalizedPath.length > 0 ? `${SITE_URL}/${normalizedPath}` : SITE_URL;

    pageData.frontmatter.head ??= [];
    pageData.frontmatter.head.push(
      ["link", { rel: "canonical", href: canonicalUrl }],
      ["meta", { property: "og:title", content: `${pageTitle} | ${SITE_NAME}` }],
      ["meta", { property: "og:url", content: canonicalUrl }],
      ["meta", { property: "og:image", content: `${SITE_URL}/cover.png` }]
    );

    if (pageData.frontmatter.description) {
      pageData.frontmatter.head.push(
        ["meta", { property: "og:description", content: pageData.frontmatter.description }],
        ["meta", { name: "description", content: pageData.frontmatter.description }]
      );
    }

    const isHome = pageData.relativePath === "index.md";
    const lastUpdated = pageData.lastUpdated
      ? new Date(pageData.lastUpdated).toISOString()
      : new Date().toISOString();

    const graph: any[] = [];

    if (isHome) {
      graph.push({
        "@type": "WebSite",
        name: SITE_NAME,
        url: SITE_URL,
        description: SITE_DESCRIPTION,
        author: { "@type": "Person", name: "Muhammad Fiaz", url: "https://github.com/muhammad-fiaz" },
      });
    }

    const authorSchema = {
      "@type": "Person",
      name: "Muhammad Fiaz",
      url: "https://muhammadfiaz.com",
      sameAs: [
        "https://github.com/muhammad-fiaz",
        "https://www.linkedin.com/in/muhammad-fiaz-",
        "https://x.com/muhammadfiaz_",
      ],
    };

    const primarySchema: Record<string, any> = {
      "@type": isHome ? "SoftwareApplication" : "TechArticle",
      name: isHome ? SITE_NAME : pageTitle,
      description: pageDescription,
      url: canonicalUrl,
      image: `${SITE_URL}/cover.png`,
      author: authorSchema,
      publisher: {
        "@type": "Organization",
        name: SITE_NAME,
        url: SITE_URL,
      },
    };

    if (isHome) {
      Object.assign(primarySchema, {
        applicationCategory: "DeveloperApplication",
        operatingSystem: "Cross-platform",
        programmingLanguage: "Zig",
        offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
        downloadUrl: "https://github.com/muhammad-fiaz/cuda.zig",
        softwareVersion: "0.0.1",
        license: "https://opensource.org/licenses/MIT",
      });
    } else {
      const pathParts = pageData.relativePath.split("/");
      const section =
        pathParts.length > 1
          ? pathParts[0].charAt(0).toUpperCase() + pathParts[0].slice(1)
          : "Documentation";
      Object.assign(primarySchema, {
        headline: pageTitle,
        articleSection: section,
        mainEntityOfPage: { "@type": "WebPage", "@id": canonicalUrl },
        datePublished: "2026-01-01T00:00:00Z",
        dateModified: lastUpdated,
      });
    }
    graph.push(primarySchema);

    // BreadcrumbList
    const breadcrumbs: any[] = [{ "@type": "ListItem", position: 1, name: "Home", item: SITE_URL }];
    if (!isHome) {
      const pathParts = pageData.relativePath.replace(/\.md$/, "").split("/");
      let currentPath = SITE_URL;
      pathParts.forEach((part: string, index: number) => {
        currentPath += `/${part}`;
        const name = part.split("-").map((s: string) => s.charAt(0).toUpperCase() + s.slice(1)).join(" ");
        breadcrumbs.push({
          "@type": "ListItem",
          position: index + 2,
          name,
          item: index === pathParts.length - 1 ? canonicalUrl : currentPath,
        });
      });
    }
    graph.push({ "@type": "BreadcrumbList", itemListElement: breadcrumbs });

    pageData.frontmatter.head.push([
      "script",
      { type: "application/ld+json" },
      JSON.stringify({ "@context": "https://schema.org", "@graph": graph }),
    ]);
  },

  themeConfig: {
    siteTitle: "cuda.zig",

    nav: [
      { text: "Home", link: "/" },
      { text: "Guide", link: "/guide/getting-started" },
      { text: "API", link: "/api/" },
      { text: "Examples", link: "/examples/" },
      {
        text: "Support",
        items: [
          { text: "💖 Sponsor", link: "https://github.com/sponsors/muhammad-fiaz" },
          { text: "☕ Donate", link: "https://pay.muhammadfiaz.com" },
        ],
      },
      { text: "GitHub", link: "https://github.com/muhammad-fiaz/cuda.zig" },
    ],

    sidebar: {
      "/guide/": [
        {
          text: "Introduction",
          items: [
            { text: "Getting Started", link: "/guide/getting-started" },
            { text: "Installation", link: "/guide/installation" },
          ],
        },
        {
          text: "Core Concepts",
          items: [
            { text: "Device Management", link: "/guide/device-management" },
            { text: "Memory Buffers", link: "/guide/memory-buffers" },
            { text: "Streams & Events", link: "/guide/streams-events" },
            { text: "Kernel Launch", link: "/guide/kernel-launch" },
            { text: "NVRTC Compilation", link: "/guide/nvrtc" },
            { text: "Multi-GPU", link: "/guide/multi-gpu" },
            { text: "CPU Fallback", link: "/guide/cpu-fallback" },
            { text: "Tensor Operations", link: "/guide/tensor-ops" },
            { text: "CUDA Allocator", link: "/guide/allocator" },
            { text: "Version Compatibility", link: "/guide/version-compat" },
          ],
        },
      ],
      "/api/": [
        {
          text: "API Reference",
          items: [
            { text: "Overview", link: "/api/" },
            { text: "Core", link: "/api/core" },
            { text: "Device", link: "/api/device" },
            { text: "Memory", link: "/api/memory" },
            { text: "Stream & Event", link: "/api/stream" },
            { text: "Kernel", link: "/api/kernel" },
            { text: "Tensor", link: "/api/tensor" },
            { text: "NVRTC", link: "/api/nvrtc" },
            { text: "Fallback", link: "/api/fallback" },
          ],
        },
      ],
      "/examples/": [
        {
          text: "Examples",
          items: [
            { text: "All Examples", link: "/examples/" },
            { text: "01 — Device Info", link: "/examples/01-device-info" },
            { text: "02 — Memory Transfer", link: "/examples/02-memory-transfer" },
            { text: "03 — Kernel Launch", link: "/examples/03-kernel-launch" },
            { text: "04 — Streams & Events", link: "/examples/04-streams-events" },
            { text: "05 — Tensor Ops", link: "/examples/05-tensor-ops" },
            { text: "06 — Multi-GPU", link: "/examples/06-multi-gpu" },
            { text: "07 — CPU Fallback", link: "/examples/07-cpu-fallback" },
            { text: "08 — Managed Memory", link: "/examples/08-managed-memory" },
            { text: "09 — NVRTC Compilation", link: "/examples/09-nvrtc-compilation" },
          ],
        },
      ],
    },

    socialLinks: [{ icon: "github", link: "https://github.com/muhammad-fiaz/cuda.zig" }],

    footer: {
      message: "Released under the MIT License.",
      copyright: "Copyright © 2026 Muhammad Fiaz",
    },

    search: { provider: "local" },

    editLink: {
      pattern: "https://github.com/muhammad-fiaz/cuda.zig/edit/main/docs/:path",
      text: "Edit this page on GitHub",
    },

    lastUpdated: {
      text: "Last updated",
      formatOptions: { dateStyle: "medium", timeStyle: "short" },
    },
  },
});
