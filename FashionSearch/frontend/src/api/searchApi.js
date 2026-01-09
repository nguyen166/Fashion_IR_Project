import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

/**
 * Get available categories for filtering
 * @returns {Promise<Array<string>>} List of categories
 */
export const getCategories = async () => {
    try {
        const response = await api.get('/categories');
        return response.data.categories || [];
    } catch (error) {
        console.error('Get categories error:', error);
        return [];
    }
};

/**
 * Get available filter options (genders, colors, seasons, usages)
 * @returns {Promise<Object>} Filter options
 */
export const getFilters = async () => {
    try {
        const response = await api.get('/filters');
        return response.data || { genders: [], colors: [], seasons: [], usages: [] };
    } catch (error) {
        console.error('Get filters error:', error);
        return { genders: [], colors: [], seasons: [], usages: [] };
    }
};

/**
 * Search by text query
 * @param {string} query - Search query text
 * @param {number} topK - Number of results to return
 * @param {string|null} category - Optional category filter
 * @returns {Promise<Array>} Search results
 */
export const searchByText = async (query, topK = 20, category = null) => {
    try {
        const body = {
            query,
            top_k: topK,
        };
        if (category) {
            body.category = category;
        }
        const response = await api.post('/search/text', body);
        return response.data;
    } catch (error) {
        console.error('Text search error:', error);
        throw error;
    }
};

/**
 * Search by image
 * @param {File} imageFile - Image file to search with
 * @param {number} topK - Number of results to return
 * @returns {Promise<Array>} Search results
 */
export const searchByImage = async (imageFile, topK = 20) => {
    try {
        const formData = new FormData();
        formData.append('file', imageFile);
        formData.append('top_k', topK.toString());

        const response = await api.post('/search/image', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    } catch (error) {
        console.error('Image search error:', error);
        throw error;
    }
};

/**
 * Refine search with relevance feedback (Rocchio Algorithm)
 * @param {Array<number>} queryVector - Original query embedding vector from previous search
 * @param {Array<number>} positiveIds - IDs of liked images
 * @param {Array<number>} negativeIds - IDs of disliked images
 * @param {number} topK - Number of results to return
 * @returns {Promise<Object>} Refined search results with new query_vector
 */
export const searchWithFeedback = async (queryVector, positiveIds, negativeIds, topK = 20) => {
    try {
        const response = await api.post('/search/feedback', {
            query_vector: queryVector,
            positive_ids: positiveIds,
            negative_ids: negativeIds,
            top_k: topK,
        });
        return response.data;
    } catch (error) {
        console.error('Feedback search error:', error);
        throw error;
    }
};

/**
 * Get full image URL from relative path
 * @param {string} relativePath - Relative image path (e.g., /images/123.jpg)
 * @returns {string} Full image URL
 */
export const getImageUrl = (relativePath) => {
    if (!relativePath) return '';
    // Remove leading slash if present to avoid double slashes
    const cleanPath = relativePath.startsWith('/') ? relativePath : `/${relativePath}`;
    return `${API_BASE_URL}${cleanPath}`;
};

export default api;
