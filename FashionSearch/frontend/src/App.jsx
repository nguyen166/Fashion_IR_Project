import React, { useState, useEffect, useCallback } from 'react';
import { Shirt } from 'lucide-react';
import SearchBar from './components/SearchBar';
import ImageGrid from './components/ImageGrid';
import RefineButton from './components/RefineButton';
import { searchByText, searchByImage, searchWithFeedback, getCategories } from './api/searchApi';

function App() {
    // Search state
    const [query, setQuery] = useState('');
    const [uploadedImage, setUploadedImage] = useState(null);
    const [results, setResults] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [hasSearched, setHasSearched] = useState(false);
    const [error, setError] = useState(null);

    // Category state
    const [categories, setCategories] = useState([]);
    const [selectedCategory, setSelectedCategory] = useState(null);

    // Feedback state
    const [positiveIds, setPositiveIds] = useState([]);
    const [negativeIds, setNegativeIds] = useState([]);

    // Query vector for refinement (from last search response)
    const [queryVector, setQueryVector] = useState(null);

    // Track last search type for refinement
    const [lastSearchType, setLastSearchType] = useState(null); // 'text' | 'image'

    // Load categories on mount
    useEffect(() => {
        const loadCategories = async () => {
            const cats = await getCategories();
            setCategories(cats);
        };
        loadCategories();
    }, []);

    // Clear selection event listener
    useEffect(() => {
        const handleClearSelection = () => {
            setPositiveIds([]);
            setNegativeIds([]);
        };
        window.addEventListener('clearSelection', handleClearSelection);
        return () => window.removeEventListener('clearSelection', handleClearSelection);
    }, []);

    // Handle text search
    const handleSearch = useCallback(async () => {
        if (!query.trim() && !uploadedImage) return;

        setIsLoading(true);
        setError(null);
        setHasSearched(true);
        // Clear previous feedback
        setPositiveIds([]);
        setNegativeIds([]);

        try {
            let searchResults;

            if (uploadedImage) {
                // Image search takes priority
                searchResults = await searchByImage(uploadedImage);
                setLastSearchType('image');
            } else {
                // Text search with optional category filter
                searchResults = await searchByText(query.trim(), 20, selectedCategory);
                setLastSearchType('text');
            }

            setResults(searchResults.results || searchResults || []);
            // Store query vector for refinement
            if (searchResults.query_vector) {
                setQueryVector(searchResults.query_vector);
            }
        } catch (err) {
            setError(err.message || 'Search failed. Please try again.');
            setResults([]);
        } finally {
            setIsLoading(false);
        }
    }, [query, uploadedImage, selectedCategory]);

    // Handle image upload
    const handleImageUpload = useCallback(async (file) => {
        setIsLoading(true);
        setError(null);
        setHasSearched(true);
        setPositiveIds([]);
        setNegativeIds([]);

        try {
            const searchResults = await searchByImage(file);
            setResults(searchResults.results || searchResults || []);
            setLastSearchType('image');
            // Store query vector for refinement
            if (searchResults.query_vector) {
                setQueryVector(searchResults.query_vector);
            }
        } catch (err) {
            setError(err.message || 'Image search failed. Please try again.');
            setResults([]);
        } finally {
            setIsLoading(false);
        }
    }, []);

    // Handle like/dislike
    const handleLike = useCallback((id) => {
        setPositiveIds(prev => {
            if (prev.includes(id)) {
                // Toggle off
                return prev.filter(i => i !== id);
            }
            // Remove from negatives and add to positives
            setNegativeIds(neg => neg.filter(i => i !== id));
            return [...prev, id];
        });
    }, []);

    const handleDislike = useCallback((id) => {
        setNegativeIds(prev => {
            if (prev.includes(id)) {
                // Toggle off
                return prev.filter(i => i !== id);
            }
            // Remove from positives and add to negatives
            setPositiveIds(pos => pos.filter(i => i !== id));
            return [...prev, id];
        });
    }, []);

    // Handle refinement (Rocchio feedback)
    const handleRefine = useCallback(async () => {
        if (positiveIds.length === 0 && negativeIds.length === 0) return;
        if (!queryVector) {
            setError('No query vector available. Please perform a search first.');
            return;
        }

        setIsLoading(true);
        setError(null);

        try {
            // Use the query vector from previous search for refinement
            const searchResults = await searchWithFeedback(
                queryVector,
                positiveIds,
                negativeIds
            );

            setResults(searchResults.results || searchResults || []);
            // Update query vector with the modified one from Rocchio
            if (searchResults.query_vector) {
                setQueryVector(searchResults.query_vector);
            }
            // Clear feedback after successful refinement
            setPositiveIds([]);
            setNegativeIds([]);
        } catch (err) {
            setError(err.message || 'Refinement failed. Please try again.');
        } finally {
            setIsLoading(false);
        }
    }, [queryVector, positiveIds, negativeIds]);

    const hasSelection = positiveIds.length > 0 || negativeIds.length > 0;

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-sky-50">
            {/* Header */}
            <header className="bg-white/80 backdrop-blur-sm border-b border-gray-100 sticky top-0 z-40">
                <div className="max-w-7xl mx-auto px-4 py-4">
                    <div className="flex items-center justify-center gap-3 mb-4">
                        <div className="p-2 bg-blue-100 rounded-xl">
                            <Shirt className="w-8 h-8 text-blue-600" />
                        </div>
                        <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-sky-600 
                           bg-clip-text text-transparent">
                            Fashion Search
                        </h1>
                    </div>

                    <SearchBar
                        query={query}
                        setQuery={setQuery}
                        onSearch={handleSearch}
                        onImageUpload={handleImageUpload}
                        uploadedImage={uploadedImage}
                        setUploadedImage={setUploadedImage}
                        isLoading={isLoading}
                        categories={categories}
                        selectedCategory={selectedCategory}
                        setSelectedCategory={setSelectedCategory}
                    />
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-7xl mx-auto px-4 py-8">
                {/* Error Message */}
                {error && (
                    <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl text-red-700">
                        {error}
                    </div>
                )}

                {/* Results Info */}
                {hasSearched && !isLoading && results.length > 0 && (
                    <div className="mb-6 flex items-center justify-between">
                        <p className="text-gray-600">
                            Found <span className="font-semibold text-blue-600">{results.length}</span> results
                            {query && <span> for "<span className="font-medium">{query}</span>"</span>}
                            {selectedCategory && <span> in <span className="font-medium">{selectedCategory}</span></span>}
                        </p>
                        {hasSelection && (
                            <p className="text-sm text-gray-500">
                                Click "Refine Search" to improve results based on your feedback
                            </p>
                        )}
                    </div>
                )}

                {/* Image Grid */}
                <ImageGrid
                    results={results}
                    positiveIds={positiveIds}
                    negativeIds={negativeIds}
                    onLike={handleLike}
                    onDislike={handleDislike}
                    isLoading={isLoading}
                    hasSearched={hasSearched}
                />
            </main>

            {/* Refine Button */}
            <RefineButton
                positiveCount={positiveIds.length}
                negativeCount={negativeIds.length}
                onRefine={handleRefine}
                isLoading={isLoading}
                isVisible={hasSelection}
            />

            {/* Footer */}
            <footer className="text-center py-6 text-gray-400 text-sm">
                Fashion Search with Rocchio Algorithm • Powered by SigLIP
            </footer>
        </div>
    );
}

export default App;
