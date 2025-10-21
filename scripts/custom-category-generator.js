// Custom category generator that sorts by top_order field
"use strict";

const pagination = require("hexo-pagination");

hexo.extend.generator.register("category", function (locals) {
  const config = this.config;
  const perPage =
    (config.category_generator && config.category_generator.per_page) ||
    config.per_page ||
    10;
  const paginationDir = config.pagination_dir || "page";

  return locals.categories.reduce((result, category) => {
    if (!category.length) return result;

    // Sort posts by top_order first, then by date
    const sortedPosts = category.posts.sort("top_order");

    const data = pagination(category.path, sortedPosts, {
      perPage: perPage,
      layout: ["category", "archive", "index"],
      format: paginationDir + "/%d/",
      data: {
        category: category.name,
      },
    });

    return result.concat(data);
  }, []);
});
