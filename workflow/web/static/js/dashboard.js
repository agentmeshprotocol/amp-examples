/**
 * Dashboard JavaScript functionality for AMP Workflow Orchestration
 */

// Global variables
let ws = null;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;

// Utility functions
function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.toggle('d-none', !show);
    }
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('liveToast');
    const toastBody = toast.querySelector('.toast-body');
    const toastHeader = toast.querySelector('.toast-header');
    
    // Set icon and color based on type
    const iconMap = {
        'success': 'fa-check-circle text-success',
        'error': 'fa-exclamation-circle text-danger',
        'warning': 'fa-exclamation-triangle text-warning',
        'info': 'fa-info-circle text-primary'
    };
    
    const icon = toastHeader.querySelector('i');
    icon.className = `fas ${iconMap[type]} me-2`;
    
    toastBody.textContent = message;
    
    const bsToast = new bootstrap.Toast(toast, {
        autohide: true,
        delay: 5000
    });
    bsToast.show();
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDateTime(dateString) {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleString();
}

function formatDuration(seconds) {
    if (!seconds || seconds < 0) return 'N/A';
    
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    } else {
        return `${secs}s`;
    }
}

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

// API helper functions
async function apiRequest(url, options = {}) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };
    
    const mergedOptions = {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...options.headers,
        },
    };
    
    try {
        const response = await fetch(url, mergedOptions);
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

// WebSocket management
function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = function() {
        console.log('WebSocket connected');
        reconnectAttempts = 0;
        updateConnectionStatus(true);
    };
    
    ws.onmessage = function(event) {
        try {
            const message = JSON.parse(event.data);
            handleWebSocketMessage(message);
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    };
    
    ws.onclose = function(event) {
        console.log('WebSocket disconnected:', event.code, event.reason);
        updateConnectionStatus(false);
        
        // Attempt to reconnect
        if (reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
            console.log(`Attempting to reconnect in ${delay}ms... (${reconnectAttempts}/${maxReconnectAttempts})`);
            setTimeout(initWebSocket, delay);
        } else {
            console.error('Max reconnection attempts reached');
            showToast('Connection lost. Please refresh the page.', 'error');
        }
    };
    
    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
        updateConnectionStatus(false);
    };
}

function updateConnectionStatus(connected) {
    const statusIcon = document.getElementById('connectionStatus');
    const statusText = document.getElementById('connectionText');
    
    if (statusIcon && statusText) {
        if (connected) {
            statusIcon.className = 'fas fa-circle text-success me-1';
            statusText.textContent = 'Connected';
        } else {
            statusIcon.className = 'fas fa-circle text-danger me-1';
            statusText.textContent = 'Disconnected';
        }
    }
}

function handleWebSocketMessage(message) {
    console.log('WebSocket message received:', message);
    
    switch (message.type) {
        case 'workflow_started':
            showToast(`Workflow started: ${message.data.instance_id}`, 'success');
            if (typeof refreshDashboard === 'function') {
                refreshDashboard();
            }
            break;
        case 'workflow_completed':
            showToast(`Workflow completed: ${message.data.instance_id}`, 'success');
            if (typeof refreshDashboard === 'function') {
                refreshDashboard();
            }
            break;
        case 'workflow_failed':
            showToast(`Workflow failed: ${message.data.instance_id}`, 'error');
            if (typeof refreshDashboard === 'function') {
                refreshDashboard();
            }
            break;
        case 'task_completed':
            if (typeof updateTaskStatus === 'function') {
                updateTaskStatus(message.data);
            }
            break;
        case 'error_event':
            showToast(`Error: ${message.data.error?.message || 'Unknown error'}`, 'error');
            break;
        default:
            console.log('Unknown message type:', message.type);
    }
}

// Form validation
function validateJSON(jsonString) {
    try {
        JSON.parse(jsonString);
        return true;
    } catch (error) {
        return false;
    }
}

function validateForm(formElement) {
    const inputs = formElement.querySelectorAll('input[required], select[required], textarea[required]');
    let isValid = true;
    
    inputs.forEach(input => {
        if (!input.value.trim()) {
            input.classList.add('is-invalid');
            isValid = false;
        } else {
            input.classList.remove('is-invalid');
            
            // Special validation for JSON fields
            if (input.dataset.type === 'json') {
                if (!validateJSON(input.value)) {
                    input.classList.add('is-invalid');
                    isValid = false;
                }
            }
        }
    });
    
    return isValid;
}

// Workflow operations
async function startWorkflow(workflowId, inputs = {}) {
    try {
        showLoading(true);
        const result = await apiRequest(`/api/workflows/${workflowId}/start`, {
            method: 'POST',
            body: JSON.stringify({
                workflow_id: workflowId,
                inputs: inputs
            })
        });
        
        showToast(`Workflow started successfully. Instance ID: ${result.instance_id}`, 'success');
        return result;
    } catch (error) {
        showToast(`Failed to start workflow: ${error.message}`, 'error');
        throw error;
    } finally {
        showLoading(false);
    }
}

async function pauseWorkflow(instanceId) {
    try {
        showLoading(true);
        await apiRequest(`/api/workflows/${instanceId}/pause`, {
            method: 'POST'
        });
        
        showToast('Workflow paused successfully', 'success');
    } catch (error) {
        showToast(`Failed to pause workflow: ${error.message}`, 'error');
        throw error;
    } finally {
        showLoading(false);
    }
}

async function resumeWorkflow(instanceId) {
    try {
        showLoading(true);
        await apiRequest(`/api/workflows/${instanceId}/resume`, {
            method: 'POST'
        });
        
        showToast('Workflow resumed successfully', 'success');
    } catch (error) {
        showToast(`Failed to resume workflow: ${error.message}`, 'error');
        throw error;
    } finally {
        showLoading(false);
    }
}

async function stopWorkflow(instanceId) {
    try {
        showLoading(true);
        await apiRequest(`/api/workflows/${instanceId}/stop`, {
            method: 'POST'
        });
        
        showToast('Workflow stopped successfully', 'success');
    } catch (error) {
        showToast(`Failed to stop workflow: ${error.message}`, 'error');
        throw error;
    } finally {
        showLoading(false);
    }
}

// File upload functionality
function setupFileUpload() {
    const fileUploadElements = document.querySelectorAll('input[type="file"]');
    
    fileUploadElements.forEach(input => {
        input.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                validateFileUpload(file);
            }
        });
    });
}

function validateFileUpload(file) {
    const maxSize = 10 * 1024 * 1024; // 10MB
    const allowedTypes = ['application/json', 'text/yaml', 'application/x-yaml'];
    
    if (file.size > maxSize) {
        showToast('File size exceeds 10MB limit', 'error');
        return false;
    }
    
    if (!allowedTypes.includes(file.type) && 
        !file.name.endsWith('.yaml') && 
        !file.name.endsWith('.yml') && 
        !file.name.endsWith('.json')) {
        showToast('Please upload a valid JSON or YAML file', 'error');
        return false;
    }
    
    return true;
}

// Data export functionality
function exportData(data, filename, type = 'json') {
    let content, mimeType, extension;
    
    switch (type) {
        case 'json':
            content = JSON.stringify(data, null, 2);
            mimeType = 'application/json';
            extension = '.json';
            break;
        case 'csv':
            content = convertToCSV(data);
            mimeType = 'text/csv';
            extension = '.csv';
            break;
        default:
            content = JSON.stringify(data, null, 2);
            mimeType = 'application/json';
            extension = '.json';
    }
    
    const blob = new Blob([content], { type: mimeType });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename + extension;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

function convertToCSV(data) {
    if (!Array.isArray(data) || data.length === 0) {
        return '';
    }
    
    const headers = Object.keys(data[0]);
    const csvContent = [
        headers.join(','),
        ...data.map(row => 
            headers.map(header => {
                const value = row[header];
                // Escape commas and quotes in CSV
                if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
                    return `"${value.replace(/"/g, '""')}"`;
                }
                return value;
            }).join(',')
        )
    ].join('\n');
    
    return csvContent;
}

// Chart utilities
function createChart(canvas, type, data, options = {}) {
    const ctx = canvas.getContext('2d');
    
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
            },
            title: {
                display: false
            }
        }
    };
    
    return new Chart(ctx, {
        type: type,
        data: data,
        options: { ...defaultOptions, ...options }
    });
}

function updateChart(chart, newData) {
    chart.data = newData;
    chart.update('none'); // No animation for real-time updates
}

// Local storage utilities
function saveToLocalStorage(key, data) {
    try {
        localStorage.setItem(key, JSON.stringify(data));
    } catch (error) {
        console.error('Failed to save to localStorage:', error);
    }
}

function loadFromLocalStorage(key, defaultValue = null) {
    try {
        const data = localStorage.getItem(key);
        return data ? JSON.parse(data) : defaultValue;
    } catch (error) {
        console.error('Failed to load from localStorage:', error);
        return defaultValue;
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard JavaScript initialized');
    
    // Initialize WebSocket connection
    initWebSocket();
    
    // Setup file upload validation
    setupFileUpload();
    
    // Setup form validation
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!validateForm(form)) {
                event.preventDefault();
                event.stopPropagation();
                showToast('Please correct the errors in the form', 'error');
            }
        });
    });
    
    // Setup keyboard shortcuts
    document.addEventListener('keydown', function(event) {
        // Ctrl+R or F5 - Refresh dashboard
        if ((event.ctrlKey && event.key === 'r') || event.key === 'F5') {
            if (typeof refreshDashboard === 'function') {
                event.preventDefault();
                refreshDashboard();
            }
        }
        
        // Escape - Close modals
        if (event.key === 'Escape') {
            const openModals = document.querySelectorAll('.modal.show');
            openModals.forEach(modal => {
                const bsModal = bootstrap.Modal.getInstance(modal);
                if (bsModal) {
                    bsModal.hide();
                }
            });
        }
    });
    
    // Setup tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Setup popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
    }
});

// Export functions for global use
window.DashboardUtils = {
    showLoading,
    showToast,
    formatBytes,
    formatDateTime,
    formatDuration,
    apiRequest,
    startWorkflow,
    pauseWorkflow,
    resumeWorkflow,
    stopWorkflow,
    exportData,
    createChart,
    updateChart,
    saveToLocalStorage,
    loadFromLocalStorage
};