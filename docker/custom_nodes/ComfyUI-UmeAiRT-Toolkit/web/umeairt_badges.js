import { app } from "../../scripts/app.js";

const style = document.createElement('style');
style.textContent = `
/* UmeAiRT Native-Style Badge */
.umeairt-premium-badge {
    background: #15151b !important; /* Solid dark background, no gradient */
    color: #ffffff !important;      /* Solid white text */
    border: 1px solid #4a5d8c !important; /* Subtle blue/purple border */
    border-radius: 4px !important;
    padding: 2px 6px !important;
    font-weight: 500 !important;
    font-size: 0.75rem !important;  /* Original size for readability */
    box-shadow: none !important;    /* No glow */
    display: inline-flex !important;
    align-items: center !important;
    gap: 5px !important;
    text-shadow: none !important;
    white-space: nowrap !important; /* Prevent text wrapping */
    flex-shrink: 0 !important;
    height: fit-content !important;
}
`;
document.head.appendChild(style);

setInterval(() => {
    // Target common badge classes used by ComfyUI and ComfyUI-Manager
    const badges = document.querySelectorAll('.cn-pack-badge, .p-tag-value, .pi-tag, span');
    
    badges.forEach(b => {
        const text = b.textContent?.trim();
        
        // Exact match to avoid styling random paragraphs
        if (text === "UmeAiRT-Toolkit" || text === "UmeAiRT-Sync" || text === "ComfyUI-UmeAiRT-Toolkit" || text === "ComfyUI-UmeAiRT-Sync") {
            
            // Skip if already upgraded
            if (b.classList.contains('umeairt-premium-badge')) return;
            
            // Apply CSS
            b.classList.add('umeairt-premium-badge');
            
            // Inject a clean star icon and the signature font "𝒰𝓂𝑒𝒜𝒾𝑅𝒯" (no suffix)
            b.innerHTML = `<span style="color: #7b8fcc; font-size: 1em; line-height: 1;">✦</span> 𝒰𝓂𝑒𝒜𝒾𝑅𝒯`;
        }
    });
}, 1000);
