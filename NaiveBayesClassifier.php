<?php

/**
 * NaiveBayesClassifier
 * 
 * A Naive Bayes text classifier that uses log probabilities and Laplace smoothing
 * to classify text documents into categories.
 */
class NaiveBayesClassifier
{
    // Valid categories for this classifier
    const VALID_CATEGORIES = [
        'Climate Change',
        'Economic Justice',
        'Immigration',
        'Reproductive Rights',
        'LGBTQIA+'
    ];
    
    // Category counts: number of documents per category
    private $categoryCounts = [];
    
    // Word counts: nested array [category][word] => count
    private $wordCounts = [];
    
    // Total number of documents trained
    private $totalDocuments = 0;
    
    // Vocabulary size (unique words across all categories)
    private $vocabularySize = 0;
    
    // Stop words to filter out during preprocessing
    private $stopWords = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
             'january', 'february', 'march', 'april', 'may', 'june', 'july',
             'august', 'september', 'october', 'november', 'december',
             'today', 'yesterday', 'tomorrow', 'week', 'month', 'year', 'time', 'said', 'says', 'according', 'new', 'state', 'states', 'country', 'countries',
                    'people', 'public', 'government', 'official', 'officials', 'agency',
                    'report', 'reports', 'law', 'bill', 'policy', 'policies', 'right', 'rights',
                    'administration', 'president', 'presidential', 'trump', 'biden', 'white house',
                    'court', 'supreme', 'justice', 'department', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'can', 'could',
                    'make', 'makes', 'made', 'take', 'takes', 'took', 'say', 'says', 'said',
                    'also', 'however', 'according', 'like', 'just', 'even', 'still', 'back', 'one', 'two', 'first', 'last', 'many', 'much', 'more', 'most', 'less', 'least',
           'very', 'really', 'quite', 'just', 'still', 'already', 'yet', 'even', 'almost',
           'actually', 'probably', 'especially', 'particularly', 'Trump', 'Biden', 'Lawmakers', 'congress', 'congressional', 'republican', 'democrat', 'man', 'woman', 'women', 'india', 'asia', 'region', 'asian', 'pacific', 'china', 'indonesia', 'japan', 'vietnam', 'europe', 'uk', 'france', 'delhi', 'california', 'utah', 'philadelphia', 'maine', 'louisiana', 'boston', 'kentucky', 'afghanistan', 'mexico', 'report', 'study', 'research', 'data', 'analysis', 'information', 'government', 'officials', 'agency', 'department', 'federal', 'security', 'administration', 'company', 'companies', 'business', 'industry', 'market', 'executive', 'director', 'group', 'said', 'says', 'according', 'called', 'found', 'show', 'shows', 'including', 'need', 'needs', 'needed', 'help', 'helping', 'provide', 'provides', 'give', 'giving', 'percent', 'million', 'billion', 'trillion', 'increase', 'higher', 'estimated', 'figure'];
    
    /**
     * Train the classifier on a single text/category pair
     * 
     * @param string $text The text content to train on
     * @param string $category The category label for this text
     * @throws Exception If category is not one of the valid categories
     */
    public function train($text, $category)
    {
        // Normalize and validate category
        $normalizedCategory = $this->normalizeCategory($category);
        
        if ($normalizedCategory === null) {
            throw new Exception("Invalid category: '$category'. Valid categories are: " . implode(', ', self::VALID_CATEGORIES));
        }
        
        // Preprocess the text to get tokens
        $tokens = $this->preprocess($text);
        
        // Initialize category if not seen before
        if (!isset($this->categoryCounts[$normalizedCategory])) {
            $this->categoryCounts[$normalizedCategory] = 0;
            $this->wordCounts[$normalizedCategory] = [];
        }
        
        // Increment category count
        $this->categoryCounts[$normalizedCategory]++;
        $this->totalDocuments++;
        
        // Count words in this document
        // Ensure all tokens are strings (numbers should be stored as strings)
        foreach ($tokens as $token) {
            $token = (string)$token; // Ensure token is a string
            if (!isset($this->wordCounts[$normalizedCategory][$token])) {
                $this->wordCounts[$normalizedCategory][$token] = 0;
            }
            $this->wordCounts[$normalizedCategory][$token]++;
        }
        
        // Recalculate vocabulary size
        $this->updateVocabularySize();
    }
    
    /**
     * Classify a new text string and return the predicted category
     * 
     * @param string $text The text to classify
     * @return string The predicted category (one of the valid categories)
     */
    public function classify($text)
    {
        if ($this->totalDocuments === 0) {
            throw new Exception("Classifier has not been trained yet.");
        }
        
        // Preprocess the text
        $tokens = $this->preprocess($text);
        
        // Calculate log probabilities for each category
        $logProbabilities = [];
        
        foreach ($this->categoryCounts as $category => $count) {
            // Log probability of the category: log(P(category))
            $logProb = log($count / $this->totalDocuments);
            
            // Add log probabilities of each word given the category: log(P(word|category))
            foreach ($tokens as $token) {
                // Ensure token is a string for lookup
                $token = (string)$token;
                // Get word count in this category (or 0 if not seen)
                $wordCount = isset($this->wordCounts[$category][$token]) 
                    ? $this->wordCounts[$category][$token] 
                    : 0;
                
                // Calculate total words in this category
                $totalWordsInCategory = array_sum($this->wordCounts[$category]);
                
                // Add-1 (Laplace) smoothing: P(word|category) = (count(word, category) + 1) / (total_words_in_category + vocabulary_size)
                $wordProbability = ($wordCount + 1) / ($totalWordsInCategory + $this->vocabularySize);
                
                // Add log probability (using log to avoid underflow)
                $logProb += log($wordProbability);
            }
            
            $logProbabilities[$category] = $logProb;
        }
        
        // Return the category with the highest log probability
        $predictedCategory = array_search(max($logProbabilities), $logProbabilities);
        
        // Safety check: ensure predicted category is valid (should always be true, but defensive programming)
        if (!in_array($predictedCategory, self::VALID_CATEGORIES)) {
            throw new Exception("Internal error: predicted category '$predictedCategory' is not valid.");
        }
        
        return $predictedCategory;
    }
    
    /**
     * Save the trained model to a JSON file
     * 
     * @param string $filePath Path to save the model file
     */
    public function saveModel($filePath)
    {
        // Ensure all stop words are strings (numbers should be stored as strings)
        $stopWordsNormalized = array_map(function($word) {
            return (string)$word;
        }, $this->stopWords);
        
        // Ensure all word count keys are strings (numbers should be stored as strings)
        $wordCountsNormalized = [];
        foreach ($this->wordCounts as $category => $words) {
            $wordCountsNormalized[$category] = [];
            foreach ($words as $word => $count) {
                $wordCountsNormalized[$category][(string)$word] = $count;
            }
        }
        
        $model = [
            'category_counts' => $this->categoryCounts,
            'word_counts' => $wordCountsNormalized,
            'total_documents' => $this->totalDocuments,
            'vocabulary_size' => $this->vocabularySize,
            'stop_words' => $stopWordsNormalized
        ];
        
        $json = json_encode($model, JSON_PRETTY_PRINT);
        
        if ($json === false) {
            throw new Exception("Failed to encode model to JSON: " . json_last_error_msg());
        }
        
        if (file_put_contents($filePath, $json) === false) {
            throw new Exception("Failed to save model to file: $filePath");
        }
    }
    
    /**
     * Load a previously saved model from a JSON file
     * 
     * @param string $filePath Path to load the model file from
     */
    public function loadModel($filePath)
    {
        if (!file_exists($filePath)) {
            throw new Exception("Model file not found: $filePath");
        }
        
        $json = file_get_contents($filePath);
        
        if ($json === false) {
            throw new Exception("Failed to read model file: $filePath");
        }
        
        $model = json_decode($json, true);
        
        if ($model === null) {
            throw new Exception("Failed to decode model JSON: " . json_last_error_msg());
        }
        
        // Validate required fields
        if (!isset($model['category_counts']) || !isset($model['word_counts']) || 
            !isset($model['total_documents']) || !isset($model['vocabulary_size'])) {
            throw new Exception("Invalid model format: missing required fields");
        }
        
        $this->categoryCounts = $model['category_counts'];
        
        // Ensure all word count keys are strings (handle numbers that might be stored as numbers)
        $this->wordCounts = [];
        foreach ($model['word_counts'] as $category => $words) {
            $this->wordCounts[$category] = [];
            foreach ($words as $word => $count) {
                $this->wordCounts[$category][(string)$word] = $count;
            }
        }
        
        $this->totalDocuments = $model['total_documents'];
        $this->vocabularySize = $model['vocabulary_size'];
        
        // Load stop words if they exist in the model (for backward compatibility)
        // Ensure all stop words are strings
        if (isset($model['stop_words'])) {
            $this->stopWords = array_map(function($word) {
                return (string)$word;
            }, $model['stop_words']);
        }
    }
    
    /**
     * Preprocess text: lowercase, remove punctuation, tokenize, remove stop words
     * 
     * @param string $text The raw text to preprocess
     * @return array Array of tokens (words)
     */
    private function preprocess($text)
    {
        // Convert to lowercase
        $text = strtolower($text);
        
        // Remove punctuation and replace with spaces
        $text = preg_replace('/[^\p{L}\p{N}\s]/u', ' ', $text);
        
        // Remove extra spaces and trim
        $text = preg_replace('/\s+/', ' ', $text);
        $text = trim($text);
        
        // Split into words
        $tokens = explode(' ', $text);
        
        // Remove stop words and empty strings
        $tokens = array_filter($tokens, function($token) {
            return !empty($token) && !in_array($token, $this->stopWords);
        });
        
        return array_values($tokens);
    }
    
    /**
     * Update the vocabulary size (unique words across all categories)
     */
    private function updateVocabularySize()
    {
        $allWords = [];
        foreach ($this->wordCounts as $category => $words) {
            $allWords = array_merge($allWords, array_keys($words));
        }
        $this->vocabularySize = count(array_unique($allWords));
    }
    
    /**
     * Get training statistics (useful for debugging/info)
     * 
     * @return array Statistics about the trained model
     */
    public function getStats()
    {
        return [
            'category_counts' => $this->categoryCounts,
            'total_documents' => $this->totalDocuments,
            'vocabulary_size' => $this->vocabularySize
        ];
    }
    
    /**
     * Normalize category name to match one of the valid categories
     * Handles case variations, underscores, hyphens, and spaces
     * 
     * @param string $category The category name to normalize
     * @return string|null The normalized category name, or null if not valid
     */
    private function normalizeCategory($category)
    {
        // Trim whitespace
        $category = trim($category);
        
        // Try exact match first (case-sensitive)
        if (in_array($category, self::VALID_CATEGORIES)) {
            return $category;
        }
        
        // Try case-insensitive match
        foreach (self::VALID_CATEGORIES as $validCategory) {
            if (strcasecmp($category, $validCategory) === 0) {
                return $validCategory;
            }
        }
        
        // Try matching with normalized format (handle underscores, hyphens, spaces)
        $normalized = strtolower($category);
        $normalized = preg_replace('/[_\-\s]+/', ' ', $normalized); // Replace underscores, hyphens, multiple spaces with single space
        $normalized = trim($normalized);
        
        foreach (self::VALID_CATEGORIES as $validCategory) {
            $validNormalized = strtolower($validCategory);
            $validNormalized = preg_replace('/[_\-\s]+/', ' ', $validNormalized);
            $validNormalized = trim($validNormalized);
            
            if ($normalized === $validNormalized) {
                return $validCategory;
            }
        }
        
        // No match found
        return null;
    }
    
    /**
     * Get list of valid categories
     * 
     * @return array List of valid category names
     */
    public static function getValidCategories()
    {
        return self::VALID_CATEGORIES;
    }
    
    /**
     * Add words that appear in multiple categories to the stop words list
     * Words that appear across many categories are less discriminative and can reduce accuracy
     * 
     * @param int $minCategories Minimum number of categories a word must appear in to be added (default: 2)
     * @return array List of words that were added to stop words
     */
    public function addMultiCategoryWordsToStopWords($minCategories = 2)
    {
        if ($this->totalDocuments === 0) {
            throw new Exception("Classifier has not been trained yet. Cannot analyze word distribution.");
        }
        
        // Count how many categories each word appears in
        $wordCategoryCount = [];
        
        foreach ($this->wordCounts as $category => $words) {
            foreach ($words as $word => $count) {
                if (!isset($wordCategoryCount[$word])) {
                    $wordCategoryCount[$word] = 0;
                }
                $wordCategoryCount[$word]++;
            }
        }
        
        // Find words that appear in multiple categories
        $wordsToAdd = [];
        foreach ($wordCategoryCount as $word => $categoryCount) {
            if ($categoryCount >= $minCategories) {
                // Ensure word is a string
                $word = (string)$word;
                // Only add if not already in stop words
                if (!in_array($word, $this->stopWords)) {
                    $wordsToAdd[] = $word;
                }
            }
        }
        
        // Add to stop words (ensure all are strings)
        $this->stopWords = array_merge($this->stopWords, $wordsToAdd);
        
        return $wordsToAdd;
    }
}

