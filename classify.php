<?php

/**
 * Classification Script for Naive Bayes Classifier
 * 
 * This script loads a trained model and classifies text provided as a command-line argument
 * or from a variable in the script.
 */

require_once 'NaiveBayesClassifier.php';

// Configuration
$modelFile = 'naive_bayes_model.json';

// Get text to classify from command line argument or use default
$textToClassify = null;

if ($argc > 1) {
    // Text provided as command-line argument
    $textToClassify = $argv[1];
} else {
    // You can also set a default text here for testing
    // $textToClassify = "New study reveals record breaking ocean temperatures";
    
    // If no argument provided, show usage
    die("Usage: php classify.php \"<text to classify>\"\n");
}

// Check if model file exists
if (!file_exists($modelFile)) {
    die("Error: Model file '$modelFile' not found. Please run train.php first.\n");
}

echo "Loading model from $modelFile...\n";

// Create classifier instance and load model
$classifier = new NaiveBayesClassifier();

try {
    $classifier->loadModel($modelFile);
    echo "Model loaded successfully!\n";
} catch (Exception $e) {
    die("Error loading model: " . $e->getMessage() . "\n");
}

// Classify the text
echo "\nClassifying text: \"$textToClassify\"\n";

try {
    $predictedCategory = $classifier->classify($textToClassify);
    echo "Predicted category: $predictedCategory\n";
} catch (Exception $e) {
    die("Error during classification: " . $e->getMessage() . "\n");
}

