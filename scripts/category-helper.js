// Custom category list helper that adds target="_blank" to links
"use strict";

hexo.extend.helper.register('list_categories_blank', function(options = {}) {
  // Access categories from hexo locals
  const categories = hexo.locals.get('categories');

  if (!categories || !categories.length) {
    return '<p>No categories available.</p>';
  }

  const showCount = options.show_count !== false;
  const className = options.class || 'category-list';

  let html = `<ul class="${className}">`;

  categories.sort('name').forEach(cat => {
    html += '<li class="category-list-item">';
    html += `<a class="category-list-link" href="${this.url_for(cat.path)}" target="_blank" rel="noopener noreferrer">`;
    html += cat.name;
    html += '</a>';
    if (showCount) {
      html += `<span class="category-list-count">${cat.length}</span>`;
    }
    html += '</li>';
  });

  html += '</ul>';

  return html;
});
