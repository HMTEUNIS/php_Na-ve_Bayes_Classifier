<?php

/**
 * Training Script for Naive Bayes Classifier
 * 
 * This script loads training data from training_data.csv, trains the classifier,
 * and saves the model to naive_bayes_model.json
 */

require_once 'NaiveBayesClassifier.php';

// Configuration
$csvFile = 'training_data.csv';
$modelFile = 'naive_bayes_model.json';

// Check if CSV file exists
if (!file_exists($csvFile)) {
    die("Error: Training data file '$csvFile' not found.\n");
}

echo "Loading training data from $csvFile...\n";
echo "Valid categories: " . implode(', ', NaiveBayesClassifier::getValidCategories()) . "\n\n";

// Create classifier instance
$classifier = new NaiveBayesClassifier();

// Open CSV file
$handle = fopen($csvFile, 'r');
if ($handle === false) {
    die("Error: Could not open CSV file '$csvFile'.\n");
}

// Read header row
$header = fgetcsv($handle, 0, ',', '"', '\\');
if ($header === false) {
    die("Error: Could not read header from CSV file.\n");
}

// Find column indices
$textIndex = array_search('TEXT', $header);
$categoryIndex = array_search('CATEGORY', $header);

if ($textIndex === false || $categoryIndex === false) {
    die("Error: CSV file must contain 'TEXT' and 'CATEGORY' columns.\n");
}

// Track training statistics
$trainingCount = 0;

// Read and train on each row
while (($row = fgetcsv($handle, 0, ',', '"', '\\')) !== false) {
    // Skip empty rows
    if (count($row) < 2 || empty($row[$textIndex]) || empty($row[$categoryIndex])) {
        continue;
    }
    
    $text = $row[$textIndex];
    $category = $row[$categoryIndex];
    
    // Train the classifier (will normalize and validate category)
    try {
        $classifier->train($text, $category);
        $trainingCount++;
    } catch (Exception $e) {
        echo "Warning: Skipping row with invalid category '$category': " . $e->getMessage() . "\n";
        continue;
    }
}

fclose($handle);

echo "Training complete!\n";
echo "Total training examples: $trainingCount\n";

// Get and display model statistics (includes normalized category counts)
$stats = $classifier->getStats();
echo "\nTraining examples per category:\n";
foreach ($stats['category_counts'] as $category => $count) {
    echo "  $category: $count\n";
}

echo "\nModel statistics:\n";
echo "  Total documents: {$stats['total_documents']}\n";
echo "  Vocabulary size: {$stats['vocabulary_size']}\n";

// Add words that appear in multiple categories to stop words
echo "\nAnalyzing word distribution across categories...\n";
try {
    $addedWords = $classifier->addMultiCategoryWordsToStopWords(2); // Words appearing in 2+ categories
    if (count($addedWords) > 0) {
        echo "Added " . count($addedWords) . " multi-category words to stop words list.\n";
        if (count($addedWords) <= 20) {
            echo "Words added: " . implode(', ', array_slice($addedWords, 0, 20)) . "\n";
        } else {
            echo "Sample words added: " . implode(', ', array_slice($addedWords, 0, 20)) . " ... (and " . (count($addedWords) - 20) . " more)\n";
        }
    } else {
        echo "No additional words needed to be added to stop words.\n";
    }
} catch (Exception $e) {
    echo "Warning: Could not analyze multi-category words: " . $e->getMessage() . "\n";
}

// Save the model
echo "\nSaving model to $modelFile...\n";
try {
    $classifier->saveModel($modelFile);
    echo "Model saved successfully!\n";
} catch (Exception $e) {
    die("Error saving model: " . $e->getMessage() . "\n");
}

