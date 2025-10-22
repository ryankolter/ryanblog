#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

/**
 * Process markdown files in a directory, adding frontmatter with title, date, and categories
 * Usage: node process-markdown.js <directory> <year-month>
 * Example: node process-markdown.js source/_posts/JS_Basic 2023-01
 */

// Parse command line arguments
const args = process.argv.slice(2);
if (args.length !== 2) {
  console.error('Usage: node process-markdown.js <directory> <year-month>');
  console.error('Example: node process-markdown.js source/_posts/JS_Basic 2023-01');
  process.exit(1);
}

const [dirPath, yearMonth] = args;

// Validate year-month format
if (!/^\d{4}-\d{2}$/.test(yearMonth)) {
  console.error('Error: year-month must be in format YYYY-MM (e.g., 2023-01)');
  process.exit(1);
}

// Check if directory exists
if (!fs.existsSync(dirPath)) {
  console.error(`Error: Directory "${dirPath}" does not exist`);
  process.exit(1);
}

// Extract category from folder name (last part of path)
const category = path.basename(dirPath).replace(/_/g, ' ');

// Get all .md files and sort them
const files = fs.readdirSync(dirPath)
  .filter(file => file.endsWith('.md'))
  .sort((a, b) => {
    // Extract leading numbers for sorting
    const numA = parseInt(a.match(/^(\d+)/)?.[1] || '0');
    const numB = parseInt(b.match(/^(\d+)/)?.[1] || '0');
    return numA - numB;
  });

if (files.length === 0) {
  console.error(`Error: No .md files found in "${dirPath}"`);
  process.exit(1);
}

console.log(`Found ${files.length} markdown files in "${dirPath}"`);
console.log(`Category: ${category}`);
console.log(`Date range: ${yearMonth}\n`);

// Generate random dates for each file within the specified month
const [year, month] = yearMonth.split('-').map(Number);
const daysInMonth = new Date(year, month, 0).getDate();
const dates = generateSortedRandomDates(year, month, daysInMonth, files.length);

// Process each file
files.forEach((file, index) => {
  const filePath = path.join(dirPath, file);
  const content = fs.readFileSync(filePath, 'utf8');

  // Extract title from filename (remove number prefix and .md extension)
  const titleMatch = file.match(/^\d+\.(.+)\.md$/);
  const title = titleMatch ? titleMatch[1] : file.replace(/\.md$/, '');

  // Check if file already has frontmatter
  if (content.startsWith('---')) {
    console.log(`⚠️  Skipping ${file} (already has frontmatter)`);
    return;
  }

  // Remove the first # heading from content
  let processedContent = content;

  // Match first heading (# Title or ## Title, etc.) and remove it
  processedContent = processedContent.replace(/^#\s+.+$/m, '').trim();

  // Extract first paragraph or section for preview (before <!--more-->)
  const lines = processedContent.split('\n');
  let previewLines = [];
  let remainingLines = [];
  let previewWordCount = 0;
  let foundEnoughContent = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    // Skip empty lines at the beginning
    if (previewLines.length === 0 && line === '') {
      continue;
    }

    // Count words (roughly) to get about 2-3 lines of content
    const words = line.split(/\s+/).filter(w => w.length > 0).length;
    previewWordCount += words;

    previewLines.push(lines[i]);

    // Stop after we have enough content (about 50-100 words or hit a section break)
    if (previewWordCount >= 50 || (line === '' && previewWordCount >= 20)) {
      foundEnoughContent = true;
      remainingLines = lines.slice(i + 1);
      break;
    }
  }

  // If we didn't find enough content, use first 3 non-empty lines
  if (!foundEnoughContent) {
    previewLines = lines.slice(0, Math.min(5, lines.length));
    remainingLines = lines.slice(Math.min(5, lines.length));
  }

  const previewContent = previewLines.join('\n').trim();
  const restContent = remainingLines.join('\n').trim();

  // Generate frontmatter with preview content before <!--more-->
  const frontmatter = `---
title: ${title}
date: ${dates[index]}
categories:
  - ${category}
---

${previewContent}

<!--more-->

${restContent}`;

  // Write back to file
  fs.writeFileSync(filePath, frontmatter, 'utf8');
  console.log(`✅ Processed: ${file}`);
  console.log(`   Title: ${title}`);
  console.log(`   Date: ${dates[index]}`);
});

console.log(`\n✨ Successfully processed ${files.length} files!`);

/**
 * Generate sorted random dates within a month
 * @param {number} year - Year
 * @param {number} month - Month (1-12)
 * @param {number} daysInMonth - Number of days in the month
 * @param {number} count - Number of dates to generate
 * @returns {string[]} Array of date strings in format YYYY-MM-DD HH:mm:ss
 */
function generateSortedRandomDates(year, month, daysInMonth, count) {
  const dates = [];

  // Generate random timestamps within the month
  const monthStart = new Date(year, month - 1, 1).getTime();
  const monthEnd = new Date(year, month, 0, 23, 59, 59).getTime();

  for (let i = 0; i < count; i++) {
    const randomTime = monthStart + Math.random() * (monthEnd - monthStart);
    dates.push(randomTime);
  }

  // Sort dates chronologically
  dates.sort((a, b) => a - b);

  // Format dates as YYYY-MM-DD HH:mm:ss
  return dates.map(timestamp => {
    const date = new Date(timestamp);
    const yyyy = date.getFullYear();
    const mm = String(date.getMonth() + 1).padStart(2, '0');
    const dd = String(date.getDate()).padStart(2, '0');
    const hh = String(date.getHours()).padStart(2, '0');
    const min = String(date.getMinutes()).padStart(2, '0');
    const ss = String(date.getSeconds()).padStart(2, '0');

    return `${yyyy}-${mm}-${dd} ${hh}:${min}:${ss}`;
  });
}
