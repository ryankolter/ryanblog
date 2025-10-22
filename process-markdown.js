#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

/**
 * Process markdown files in a directory (including subdirectories), adding frontmatter
 * Can be run multiple times - updates top_order for existing files and processes new ones
 * Usage: node process-markdown.js <directory> <year-month>
 * Example: node process-markdown.js source/_posts/JavaScript 2023-01
 */

// Parse command line arguments
const args = process.argv.slice(2);
if (args.length !== 2) {
  console.error('Usage: node process-markdown.js <directory> <year-month>');
  console.error('Example: node process-markdown.js source/_posts/JavaScript 2023-01');
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

// Extract category from the main folder name (the target directory basename)
const category = path.basename(dirPath).replace(/_/g, ' ');

/**
 * Recursively get all .md files from directory and subdirectories
 */
function getAllMarkdownFiles(dir, fileList = []) {
  const files = fs.readdirSync(dir);

  files.forEach(file => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      // Recursively search subdirectories
      getAllMarkdownFiles(filePath, fileList);
    } else if (file.endsWith('.md')) {
      fileList.push(filePath);
    }
  });

  return fileList;
}

/**
 * Parse frontmatter from content
 */
function parseFrontmatter(content) {
  if (!content.startsWith('---')) {
    return { hasFrontmatter: false, frontmatter: null, bodyContent: content };
  }

  const lines = content.split('\n');
  let endIndex = -1;

  // Find the closing ---
  for (let i = 1; i < lines.length; i++) {
    if (lines[i].trim() === '---') {
      endIndex = i;
      break;
    }
  }

  if (endIndex === -1) {
    return { hasFrontmatter: false, frontmatter: null, bodyContent: content };
  }

  const frontmatterLines = lines.slice(1, endIndex);
  const bodyContent = lines.slice(endIndex + 1).join('\n');

  // Parse frontmatter into object
  const frontmatter = {};
  frontmatterLines.forEach(line => {
    const match = line.match(/^(\w+):\s*(.*)$/);
    if (match) {
      const [, key, value] = match;
      frontmatter[key] = value;
    }
  });

  return { hasFrontmatter: true, frontmatter, bodyContent };
}

// Get all .md files recursively and sort them by their full path
const allFiles = getAllMarkdownFiles(dirPath);

/**
 * Extract numeric prefix from a path component (folder or file name)
 */
function getNumericPrefix(name) {
  const match = name.match(/^(\d+)/);
  return match ? parseInt(match[1]) : 0;
}

// Sort files by folder hierarchy first, then by file name
allFiles.sort((a, b) => {
  // Get relative paths from the base directory
  const relPathA = path.relative(dirPath, a);
  const relPathB = path.relative(dirPath, b);

  // Split paths into components (folders and file)
  const partsA = relPathA.split(path.sep);
  const partsB = relPathB.split(path.sep);

  // Compare each level of the path hierarchy
  const minLength = Math.min(partsA.length, partsB.length);

  for (let i = 0; i < minLength; i++) {
    const numA = getNumericPrefix(partsA[i]);
    const numB = getNumericPrefix(partsB[i]);

    if (numA !== numB) {
      return numA - numB;
    }

    // If numeric prefixes are equal, compare lexicographically
    if (partsA[i] !== partsB[i]) {
      return partsA[i].localeCompare(partsB[i]);
    }
  }

  // If all parts are equal up to minLength, shorter path comes first
  return partsA.length - partsB.length;
});

if (allFiles.length === 0) {
  console.error(`Error: No .md files found in "${dirPath}"`);
  process.exit(1);
}

console.log(`Found ${allFiles.length} markdown files in "${dirPath}" (including subdirectories)`);
console.log(`Category: ${category}`);
console.log(`Date range: ${yearMonth}\n`);

// Generate dates for ALL files (both new and existing) to maintain chronological order
const [year, month] = yearMonth.split('-').map(Number);

// Generate sorted dates for all files to maintain order
const dates = generateSortedRandomDates(year, month, allFiles.length);

// Process each file
allFiles.forEach((filePath, index) => {
  const content = fs.readFileSync(filePath, 'utf8');
  const fileName = path.basename(filePath);
  const relativePath = path.relative(dirPath, filePath);

  // Extract title from filename (remove number prefix and .md extension)
  // Handles formats like: 00.title.md, 0010.title.md -> title
  const titleMatch = fileName.match(/^\d+\.(.+)\.md$/);
  let title = titleMatch ? titleMatch[1] : fileName.replace(/\.md$/, '');

  // Also remove leading digits and dot from the title itself (e.g., "00.something" -> "something")
  title = title.replace(/^\d+\.\s*/, '');

  // Calculate top_order (10, 20, 30, ...)
  const topOrder = (index + 1) * 10;

  const { hasFrontmatter, frontmatter, bodyContent } = parseFrontmatter(content);

  // Use date from sorted dates array to maintain chronological order
  const dateToUse = dates[index];

  if (hasFrontmatter) {
    // Update existing file - update both top_order and date to maintain order
    // Reconstruct frontmatter with updated top_order and date
    const updatedFrontmatter = `---
title: ${frontmatter.title || title}
date: ${dateToUse}
categories:
  - ${frontmatter.categories || category}
top_order: ${topOrder}
---
${bodyContent}`;

    fs.writeFileSync(filePath, updatedFrontmatter, 'utf8');
    console.log(`🔄 Updated: ${relativePath}`);
    console.log(`   Date: ${dateToUse} (updated)`);
    console.log(`   Top Order: ${topOrder} (updated)`);
    return;
  }

  // New file - process completely

  // Remove the first # heading from content
  let processedContent = bodyContent;

  // Match first heading (# Title or ## Title, etc.) and remove it
  processedContent = processedContent.replace(/^#\s+.+$/m, '').trim();

  // Extract first paragraph or section for preview (before <!--more-->)
  const lines = processedContent.split('\n');
  let previewEndIndex = -1;
  let previewWordCount = 0;
  let startedContent = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    // Skip empty lines at the beginning
    if (!startedContent && line === '') {
      continue;
    }

    if (line !== '') {
      startedContent = true;
    }

    // Count words (roughly) to get about 2-3 lines of content
    const words = line.split(/\s+/).filter(w => w.length > 0).length;
    previewWordCount += words;

    // Stop after we have enough content (about 50-100 words or hit a section break)
    if (previewWordCount >= 50 || (line === '' && previewWordCount >= 20)) {
      previewEndIndex = i;
      break;
    }
  }

  // If we didn't find enough content, use first 5 lines
  if (previewEndIndex === -1) {
    previewEndIndex = Math.min(5, lines.length - 1);
  }

  const previewContent = lines.slice(0, previewEndIndex + 1).join('\n').trim();
  const restContent = lines.slice(previewEndIndex + 1).join('\n').trim();

  // Generate frontmatter with preview content before <!--more-->
  const newContent = `---
title: ${title}
date: ${dateToUse}
categories:
  - ${category}
top_order: ${topOrder}
---

${previewContent}

<!--more-->

${restContent}`;

  // Write back to file
  fs.writeFileSync(filePath, newContent, 'utf8');
  console.log(`✅ Processed: ${relativePath}`);
  console.log(`   Title: ${title}`);
  console.log(`   Date: ${dateToUse}`);
  console.log(`   Top Order: ${topOrder}`);
});

console.log(`\n✨ Successfully processed ${allFiles.length} files!`);

/**
 * Generate sorted random dates within a month (night time only: 19:00 - 23:59)
 * @param {number} year - Year
 * @param {number} month - Month (1-12)
 * @param {number} count - Number of dates to generate
 * @returns {string[]} Array of date strings in format YYYY-MM-DD HH:mm:ss
 */
function generateSortedRandomDates(year, month, count) {
  const dates = [];
  const daysInMonthActual = new Date(year, month, 0).getDate();

  // Generate random timestamps for each file
  for (let i = 0; i < count; i++) {
    // Random day in the month (1 to daysInMonth)
    const day = Math.floor(Math.random() * daysInMonthActual) + 1;

    // Random hour between 19:00 and 23:59 (19-23)
    const hour = Math.floor(Math.random() * 5) + 19; // 19, 20, 21, 22, 23

    // Random minute (0-59)
    const minute = Math.floor(Math.random() * 60);

    // Random second (0-59)
    const second = Math.floor(Math.random() * 60);

    const timestamp = new Date(year, month - 1, day, hour, minute, second).getTime();
    dates.push(timestamp);
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
