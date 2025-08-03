/**
 * Knowledge Base System - Main JavaScript
 */

// Global variables
let systemStatus = 'unknown';
let agentStatuses = {};

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    updateSystemStatus();
    
    // Set up periodic updates
    setInterval(updateSystemStatus, 30000); // Every 30 seconds
});

/**
 * Initialize the application
 */
function initializeApp() {
    // Set up global event listeners
    setupGlobalEventListeners();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Check for URL parameters
    handleUrlParameters();
}

/**
 * Set up global event listeners
 */
function setupGlobalEventListeners() {
    // Handle navigation highlighting
    highlightCurrentPage();
    
    // Handle file upload drag and drop
    setupDragAndDrop();
    
    // Handle keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);
}

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Handle URL parameters
 */
function handleUrlParameters() {
    const urlParams = new URLSearchParams(window.location.search);
    
    // Auto-focus search if on search page
    if (window.location.pathname === '/search' && document.getElementById('search-query')) {
        document.getElementById('search-query').focus();
    }
}

/**
 * Highlight current page in navigation
 */
function highlightCurrentPage() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href === currentPath || (currentPath !== '/' && href !== '/' && currentPath.startsWith(href))) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

/**
 * Set up drag and drop for file uploads
 */
function setupDragAndDrop() {
    const uploadArea = document.querySelector('.upload-area');
    if (!uploadArea) return;
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });
    
    uploadArea.addEventListener('drop', handleDrop, false);
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight(e) {
        uploadArea.classList.add('dragover');
    }
    
    function unhighlight(e) {
        uploadArea.classList.remove('dragover');
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }
}

/**
 * Handle dropped files
 */
function handleFiles(files) {
    Array.from(files).forEach(uploadFile);
}

/**
 * Upload a file
 */
function uploadFile(file) {
    console.log('Uploading file:', file.name);
    // Implementation would depend on the upload form
}

/**
 * Handle keyboard shortcuts
 */
function handleKeyboardShortcuts(e) {
    // Ctrl/Cmd + K for search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.getElementById('search-query') || document.getElementById('quick-search-input');
        if (searchInput) {
            searchInput.focus();
            searchInput.select();
        }
    }
    
    // Escape to clear modals
    if (e.key === 'Escape') {
        const modals = document.querySelectorAll('.modal.show');
        modals.forEach(modal => {
            const modalInstance = bootstrap.Modal.getInstance(modal);
            if (modalInstance) {
                modalInstance.hide();
            }
        });
    }
}

/**
 * Update system status
 */
async function updateSystemStatus() {
    try {
        const response = await fetch('/api/system/status');
        const data = await response.json();
        
        if (data.success) {
            systemStatus = data.system_status;
            agentStatuses = data.agents;
            
            updateSystemStatusDisplay();
            updateAgentStatusDisplay();
        } else {
            systemStatus = 'error';
            updateSystemStatusDisplay();
        }
    } catch (error) {
        console.error('Failed to update system status:', error);
        systemStatus = 'error';
        updateSystemStatusDisplay();
    }
}

/**
 * Update system status display
 */
function updateSystemStatusDisplay() {
    const statusElement = document.getElementById('system-status');
    if (!statusElement) return;
    
    statusElement.className = 'badge';
    
    switch (systemStatus) {
        case 'operational':
            statusElement.classList.add('bg-success');
            statusElement.textContent = 'Operational';
            break;
        case 'degraded':
            statusElement.classList.add('bg-warning');
            statusElement.textContent = 'Degraded';
            break;
        case 'error':
            statusElement.classList.add('bg-danger');
            statusElement.textContent = 'Error';
            break;
        default:
            statusElement.classList.add('bg-secondary');
            statusElement.textContent = 'Unknown';
    }
}

/**
 * Update agent status display
 */
function updateAgentStatusDisplay() {
    const activeAgentsElement = document.getElementById('active-agents');
    if (activeAgentsElement && agentStatuses) {
        const activeCount = Object.values(agentStatuses).filter(status => status === 'online').length;
        const totalCount = Object.keys(agentStatuses).length;
        activeAgentsElement.textContent = `${activeCount}/${totalCount}`;
    }
}

/**
 * Show a toast notification
 */
function showToast(message, type = 'info', duration = 5000) {
    const toastContainer = getOrCreateToastContainer();
    
    const toastId = 'toast-' + Date.now();
    const toast = document.createElement('div');
    toast.id = toastId;
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">${message}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" 
                    data-bs-dismiss="toast"></button>
        </div>
    `;
    
    toastContainer.appendChild(toast);
    
    const bsToast = new bootstrap.Toast(toast, { delay: duration });
    bsToast.show();
    
    // Remove from DOM after hiding
    toast.addEventListener('hidden.bs.toast', () => {
        toast.remove();
    });
}

/**
 * Get or create toast container
 */
function getOrCreateToastContainer() {
    let container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
    }
    return container;
}

/**
 * Show loading state
 */
function showLoading(elementId, show = true) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    if (show) {
        const loadingOverlay = document.createElement('div');
        loadingOverlay.className = 'loading-overlay';
        loadingOverlay.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary mb-2" role="status"></div>
                <div class="small">Loading...</div>
            </div>
        `;
        element.style.position = 'relative';
        element.appendChild(loadingOverlay);
    } else {
        const overlay = element.querySelector('.loading-overlay');
        if (overlay) {
            overlay.remove();
        }
    }
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Format duration
 */
function formatDuration(seconds) {
    if (seconds < 60) {
        return `${seconds.toFixed(1)}s`;
    } else if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}m ${remainingSeconds}s`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
    }
}

/**
 * Format quality score
 */
function formatQualityScore(score, includeText = true) {
    const percentage = (score * 100).toFixed(1);
    let className = 'quality-fair';
    let text = 'Fair';
    
    if (score >= 0.8) {
        className = 'quality-excellent';
        text = 'Excellent';
    } else if (score >= 0.6) {
        className = 'quality-good';
        text = 'Good';
    } else if (score < 0.4) {
        className = 'quality-poor';
        text = 'Poor';
    }
    
    if (includeText) {
        return `<span class="${className}">${percentage}% (${text})</span>`;
    } else {
        return `<span class="${className}">${percentage}%</span>`;
    }
}

/**
 * Debounce function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Throttle function
 */
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * Copy text to clipboard
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showToast('Copied to clipboard', 'success');
    } catch (err) {
        console.error('Failed to copy text: ', err);
        showToast('Failed to copy to clipboard', 'danger');
    }
}

/**
 * Download data as file
 */
function downloadAsFile(data, filename, type = 'application/json') {
    const blob = new Blob([data], { type });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * Validate form inputs
 */
function validateForm(formElement) {
    const inputs = formElement.querySelectorAll('input[required], textarea[required], select[required]');
    let isValid = true;
    
    inputs.forEach(input => {
        if (!input.value.trim()) {
            input.classList.add('is-invalid');
            isValid = false;
        } else {
            input.classList.remove('is-invalid');
            input.classList.add('is-valid');
        }
    });
    
    return isValid;
}

/**
 * Clear form validation
 */
function clearFormValidation(formElement) {
    const inputs = formElement.querySelectorAll('.is-invalid, .is-valid');
    inputs.forEach(input => {
        input.classList.remove('is-invalid', 'is-valid');
    });
}

/**
 * Animate element entrance
 */
function animateIn(element, animationClass = 'fade-in') {
    element.classList.add(animationClass);
    
    // Remove animation class after animation completes
    element.addEventListener('animationend', () => {
        element.classList.remove(animationClass);
    }, { once: true });
}

/**
 * Scroll to element smoothly
 */
function scrollToElement(elementId, offset = 0) {
    const element = document.getElementById(elementId);
    if (element) {
        const y = element.offsetTop - offset;
        window.scrollTo({ top: y, behavior: 'smooth' });
    }
}

/**
 * Utility function to escape HTML
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Utility function to format dates
 */
function formatDate(dateString, includeTime = false) {
    const date = new Date(dateString);
    const options = {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    };
    
    if (includeTime) {
        options.hour = '2-digit';
        options.minute = '2-digit';
    }
    
    return date.toLocaleDateString('en-US', options);
}

/**
 * Get relative time string
 */
function getRelativeTime(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diffInSeconds = Math.floor((now - date) / 1000);
    
    if (diffInSeconds < 60) return 'just now';
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)} minutes ago`;
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)} hours ago`;
    if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)} days ago`;
    
    return formatDate(dateString);
}

// Export functions for use in other scripts
window.KnowledgeBase = {
    showToast,
    showLoading,
    formatFileSize,
    formatDuration,
    formatQualityScore,
    copyToClipboard,
    downloadAsFile,
    validateForm,
    clearFormValidation,
    animateIn,
    scrollToElement,
    escapeHtml,
    formatDate,
    getRelativeTime,
    debounce,
    throttle
};