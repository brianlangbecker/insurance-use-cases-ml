import { initApp } from './app.js';

// This bundle is meant for the browser. Guard so tooling that imports modules in Node
// (without a DOM/window) doesn't immediately throw.
if (typeof window !== 'undefined') {
    initApp();
}
