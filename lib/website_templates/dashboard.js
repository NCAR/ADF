/* ============================================================
   CESM Dashboard Javascript
   ============================================================ */


/* ============================================================
   Global State
   ============================================================ */

let dashboardData = [];

let currentCaseIndex = 0;

let currentCase = null;

let currentImages = [];

let currentImageIndex = 0;

/*
   "summary" or "individual" - which images the View
   dropdown/thumbnail list shows for a plot type that has
   a panel breakdown (see getPanelGroupKey below). Global
   rather than per-case, so it persists across Case changes.
*/
let currentViewMode = "summary";


/*
   These contain INDIVIDUAL image instances.

   Example:

       "0:0"
       "0:2"
       "1:1"

   means:

       Case 0 / Image 0
       Case 0 / Image 2
       Case 1 / Image 1
*/
let mosaicSelection = new Set();

let sliderPlotTypes = new Set();

let sliderCaseSelection = new Set();


/*
   Display/playback order of the case checkboxes on
   the Slider tab. Holds case indices as strings.
   Reordered by drag-and-drop; getSliderImages() walks
   this array (filtered by sliderCaseSelection) so users
   can control the order images play in.
*/
let sliderCaseOrder = [];


/*
   Current image being displayed in the Slider.
*/
let currentSliderSelectionIndex = 0;


/*
   setInterval handle for "Play" on the Slider tab.
   null when playback is stopped.
*/
let sliderPlaybackTimer = null;

const sliderPlaybackIntervalMs = 1200;


/*
   Case index currently being dragged in the Slider's
   case list (for reordering).
*/
let draggedSliderCaseKey = null;


/* ============================================================
   Initialization
   ============================================================ */

document.addEventListener(
    "DOMContentLoaded",
    initializeDashboard
);


async function initializeDashboard() {

    setupTabs();

    setupButtons();

    await loadManifest();


    if (!dashboardData.length) {

        console.error(
            "No dashboard cases found."
        );

        return;

    }


    currentCaseIndex = 0;

    currentCase =
        dashboardData[0];

    currentImages =
        currentCase.urls || [];

    currentImageIndex = 0;


    buildCaseList();

    buildThumbnails();

    buildPanelSelector();

    updateMainImage();

    buildMosaicSelector();

    buildMosaic();

    buildSlider();

    buildInteractiveTabs();

    updateCaseSelection();

}


/* ============================================================
   Global Buttons
   ============================================================ */

function setupButtons() {

    const saveMosaicButton =
        document.getElementById(
            "save-mosaic"
        );


    if (saveMosaicButton) {

        saveMosaicButton.addEventListener(
            "click",
            saveMosaicPNG
        );

    }

}


/* ============================================================
   Manifest
   ============================================================ */

async function loadManifest() {

    try {

        const response =
            await fetch("manifest.json");


        if (!response.ok) {

            throw new Error(
                "Could not load manifest JSON file."
            );

        }


        dashboardData =
            await response.json();


        if (!Array.isArray(dashboardData)) {

            throw new Error(
                "Manifest JSON file must contain an array."
            );

        }


        console.log(
            "Loaded manifest:",
            dashboardData
        );

    }

    catch(error) {

        console.error(
            "Error loading manifest:",
            error
        );

    }

}


/* ============================================================
   Image IDs
   ============================================================ */


/*
   The key that makes two images "the same kind of plot" for
   the Slider: same plot type, and - for a plot type that has
   a panel breakdown - the same panel too. Images without a
   "panel" (plot types with no summary/component structure)
   fall back to grouping by type alone, same as before.
*/
function getPanelGroupKey(image) {

    return image.panel
        ? `${image.type}::${image.panel}`
        : image.type;

}


/*
   Create a unique ID for an individual image.
*/
function getImageID(
    caseIndex,
    imageIndex
) {

    return `${caseIndex}:${imageIndex}`;

}


/*
   Convert an image ID back to indexes.
*/
function parseImageID(id) {

    const parts =
        id.split(":");


    return {

        caseIndex:
            Number(parts[0]),

        imageIndex:
            Number(parts[1])

    };

}


/*
   Get an image object from an image ID.
*/
function getImageFromID(id) {

    const {
        caseIndex,
        imageIndex
    } = parseImageID(id);


    const caseData =
        dashboardData[caseIndex];


    if (!caseData) {

        return null;

    }


    const image =
        caseData.urls?.[imageIndex];


    if (!image) {

        return null;

    }


    return {

        id: id,

        caseIndex: caseIndex,

        imageIndex: imageIndex,

        caseData: caseData,

        image: image

    };

}


/* ============================================================
   Case List
   ============================================================ */

function buildCaseList() {

    const container =
        document.getElementById(
            "case-container"
        );


    if (!container) {

        return;

    }


    container.innerHTML = "";


    const select =
        document.createElement(
            "select"
        );


    select.className =
        "control-button case-select";


    select.id =
        "case-select";


    dashboardData.forEach(
        (caseData, index) => {

            const option =
                document.createElement(
                    "option"
                );


            option.value =
                index;


            option.textContent =
                caseData.case;


            select.appendChild(
                option
            );

        }
    );


    select.onchange =
        () => {

            selectCase(
                Number(
                    select.value
                )
            );

        };


    container.appendChild(
        select
    );


    updateCaseSelection();

}


/* ============================================================
   Select Case
   ============================================================ */

function selectCase(index) {

    if (
        index < 0 ||
        index >= dashboardData.length
    ) {

        return;

    }


    /*
       Remember the current image TYPE and, if it has
       one, its PANEL (e.g. "summary", "open_cell").

       This is what makes:

           Nd LWC PDF
           Dropsonde Comparison: Open-Cell Bias

       stay selected while changing cases.
    */

    let selectedImageType = null;

    let selectedPanel = null;


    if (
        currentImages.length &&
        currentImages[currentImageIndex]
    ) {

        selectedImageType =
            currentImages[currentImageIndex].type;

        selectedPanel =
            currentImages[currentImageIndex].panel ||
            null;

    }


    currentCaseIndex =
        index;


    currentCase =
        dashboardData[index];


    currentImages =
        currentCase.urls || [];


    /*
       Find the same image type (and, if there was
       one, the same panel) in the newly selected case.
    */

    if (selectedImageType) {

        let matchingIndex = -1;


        if (selectedPanel) {

            matchingIndex =
                currentImages.findIndex(
                    image =>
                        image.type === selectedImageType &&
                        image.panel === selectedPanel
                );

        }


        if (matchingIndex < 0) {

            matchingIndex =
                currentImages.findIndex(
                    image =>
                        image.type ===
                        selectedImageType
                );

        }


        currentImageIndex =
            matchingIndex >= 0
                ? matchingIndex
                : 0;

    }

    else {

        currentImageIndex = 0;

    }


    buildThumbnails();

    buildPanelSelector();

    updateMainImage();

    updateCaseSelection();


    /*
       The sidebar changed case, but the
       Mosaic selection itself does NOT change.

       It is intentionally independent.
    */

    buildMosaicSelector();

    buildMosaic();

    buildSlider();

}


/* ============================================================
   Panel Selector (Summary vs. component panels)
   ============================================================ */

/*
   Some plot types (e.g. Dropsonde Comparison) register a
   multi-panel "summary" image alongside its individual
   component panels (Open-Cell, Stratocumulus, ...), tagged
   with "panel"/"panel_label" in the manifest. When the
   currently selected image's type has this breakdown, this
   builds a fixed Summary/Individual Plots toggle: it doesn't
   pick a single image, it sets currentViewMode, which
   buildThumbnails() uses to decide what to list (see there).
   Plot types without a panel breakdown simply have no "panel"
   key on their image entries, so the selector stays hidden.
*/
function buildPanelSelector() {

    const heading =
        document.getElementById(
            "panel-heading"
        );

    const container =
        document.getElementById(
            "panel-container"
        );


    if (!container) {

        return;

    }


    container.innerHTML = "";


    const currentImage =
        currentImages[currentImageIndex];


    if (!currentImage) {

        if (heading) heading.style.display = "none";

        return;

    }


    const hasPanelData =
        currentImages.some(
            image => image.panel
        );


    if (!hasPanelData) {

        if (heading) heading.style.display = "none";

        return;

    }


    if (heading) heading.style.display = "";


    const select =
        document.createElement(
            "select"
        );


    select.className =
        "control-button case-select";


    select.id =
        "panel-select";


    const viewOptions = [
        { value: "summary",    label: "Summary" },
        { value: "individual", label: "Individual Plots" },
    ];

    viewOptions.forEach(
        opt => {

            const option =
                document.createElement(
                    "option"
                );


            option.value =
                opt.value;


            option.textContent =
                opt.label;


            select.appendChild(
                option
            );

        }
    );


    select.value =
        currentViewMode;


    select.onchange =
        () => {

            currentViewMode =
                select.value;


            const entries =
                buildThumbnails();


            /*
               Jump the main image to the first thumbnail
               now visible under the new View mode, so it
               isn't left showing something that's no
               longer in the (now-changed) list.
            */

            if (entries && entries.length) {

                selectImageByID(
                    entries[0].caseIndex,
                    entries[0].imageIndex
                );

            }

        };


    container.appendChild(
        select
    );

}


/* ============================================================
   Sidebar Thumbnails
   ============================================================ */

/*
   Builds the "Images" thumbnail list, and returns the list of
   {caseIndex, imageIndex, image, caseData} entries it rendered
   (used by the View dropdown to jump to the first entry after
   a mode change).

   Always scoped to the currently selected Case (never pooled
   across cases - the Case dropdown is what picks the case).
   When that case has any panel-tagged images (see
   getPanelGroupKey), the list is filtered by currentViewMode:
   "summary" shows just each plot type's one Summary image (e.g.
   3, one per plot type), "individual" shows every non-Summary
   component panel across every plot type the case has (e.g.
   3 x 4 = 12) - deliberately NOT split by plot type, so e.g.
   Dropsonde/Nd-LWC/Sensitivity panels all show together. Cases
   with no panel-tagged images at all fall back to showing
   everything, same as before panels existed.
*/
function buildThumbnails() {

    const container =
        document.getElementById(
            "thumbnail-container"
        );


    if (!container) {

        return [];

    }


    container.innerHTML = "";


    const hasPanelData =
        currentImages.some(
            image => image.panel
        );


    const entries =
        currentImages
            .map(
                (image, imageIndex) => ({
                    caseIndex: currentCaseIndex,
                    imageIndex,
                    image,
                    caseData: currentCase,
                })
            )
            .filter(
                ({ image }) => {

                    if (!hasPanelData) {

                        return true;

                    }


                    if (!image.panel) {

                        return false;

                    }


                    const isSummary =
                        image.panel === "summary";


                    return currentViewMode === "summary"
                        ? isSummary
                        : !isSummary;

                }
            );


    entries.forEach(
        ({ caseIndex, imageIndex, image: item }) => {

            const imageID =
                getImageID(
                    caseIndex,
                    imageIndex
                );


            const wrapper =
                document.createElement(
                    "div"
                );


            wrapper.className =
                "thumbnail-wrapper";


            /*
               Thumbnail
            */

            const img =
                document.createElement(
                    "img"
                );


            img.src =
                item.url;


            img.alt =
                item.type;


            img.title = hasPanelData
                ? `${item.type}: ${item.panel_label || item.panel}`
                : `${item.category}: ${item.type}`;


            img.className =
                "thumbnail";


            img.dataset.imageId =
                imageID;


            img.onclick =
                () => {

                    selectImageByID(
                        caseIndex,
                        imageIndex
                    );

                };


            wrapper.appendChild(
                img
            );


            /*
               Image name
            */

            const label =
                document.createElement(
                    "div"
                );


            label.className =
                "thumbnail-label";


            label.textContent = hasPanelData
                ? `${item.type}: ${item.panel_label || item.panel}`
                : item.type;


            wrapper.appendChild(
                label
            );


            /*
               Slider / Mosaic controls
            */

            const controls =
                document.createElement(
                    "div"
                );


            controls.className =
                "image-selection-controls";


            /*
            Slider
            */

            const sliderLabel =
                document.createElement(
                    "label"
                );


            const sliderCheckbox =
                document.createElement(
                    "input"
                );


            sliderCheckbox.type =
                "checkbox";


            sliderCheckbox.checked =
                sliderPlotTypes.has(
                    getPanelGroupKey(item)
                );


            sliderCheckbox.addEventListener(
                "click",
                event => {

                    event.stopPropagation();

                }
            );


            sliderCheckbox.addEventListener(
                "change",
                () => {

                    const groupKey =
                        getPanelGroupKey(item);


                    if (sliderCheckbox.checked) {

                        /*
                        The Slider uses ONE plot type
                        (and, if applicable, panel) at
                        a time.

                        If another group was already
                        selected, remove it.
                        */

                        sliderPlotTypes.clear();

                        sliderPlotTypes.add(
                            groupKey
                        );


                        /*
                        Automatically select every case
                        that contains this same group
                        (type + panel, when there is one),
                        so the Slider only ever compares
                        like-for-like images.
                        */

                        sliderCaseSelection.clear();

                        sliderCaseOrder = [];


                        dashboardData.forEach(
                            (otherCaseData, otherCaseIndex) => {

                                const hasGroup =
                                    otherCaseData.urls?.some(
                                        image =>
                                            getPanelGroupKey(image) ===
                                            groupKey
                                    );


                                if (hasGroup) {

                                    sliderCaseSelection.add(
                                        String(otherCaseIndex)
                                    );

                                    sliderCaseOrder.push(
                                        String(otherCaseIndex)
                                    );

                                }

                            }
                        );

                    }

                    else {

                        /*
                        Remove this group from
                        the Slider.
                        */

                        sliderPlotTypes.delete(
                            groupKey
                        );


                        /*
                        Clear the case selections because
                        there is no active group.
                        */

                        sliderCaseSelection.clear();

                        sliderCaseOrder = [];

                    }


                    currentSliderSelectionIndex = 0;


                    stopSliderPlayback();


                    buildSlider();

                    buildThumbnails();

                }
            );


            sliderLabel.appendChild(
                sliderCheckbox
            );


            sliderLabel.appendChild(
                document.createTextNode(
                    " Slider"
                )
            );


            controls.appendChild(
                sliderLabel
            );


            /*
               Mosaic
            */

            const mosaicLabel =
                document.createElement(
                    "label"
                );


            const mosaicCheckbox =
                document.createElement(
                    "input"
                );


            mosaicCheckbox.type =
                "checkbox";


            mosaicCheckbox.checked =
                mosaicSelection.has(
                    imageID
                );


            mosaicCheckbox.addEventListener(
                "click",
                event => {

                    event.stopPropagation();

                }
            );


            mosaicCheckbox.addEventListener(
                "change",
                () => {

                    if (
                        mosaicCheckbox.checked
                    ) {

                        mosaicSelection.add(
                            imageID
                        );

                    }

                    else {

                        mosaicSelection.delete(
                            imageID
                        );

                    }


                    buildMosaicSelector();

                    buildMosaic();

                }
            );


            mosaicLabel.appendChild(
                mosaicCheckbox
            );


            mosaicLabel.appendChild(
                document.createTextNode(
                    " Mosaic"
                )
            );


            controls.appendChild(
                mosaicLabel
            );


            wrapper.appendChild(
                controls
            );


            container.appendChild(
                wrapper
            );

        }
    );


    updateThumbnailSelection();


    return entries;

}


/* ============================================================
   Main Image
   ============================================================ */

/*
   Select an image by its owning case, not just an index into
   currentImages - needed because the thumbnail list can be
   pooled across cases (see buildThumbnails). Also re-syncs the
   Case dropdown, since a pooled thumbnail may belong to a case
   other than the one currently selected there.
*/
function selectImageByID(caseIndex, imageIndex) {

    const caseData =
        dashboardData[caseIndex];


    if (!caseData) {

        return;

    }


    const images =
        caseData.urls || [];


    if (
        imageIndex < 0 ||
        imageIndex >= images.length
    ) {

        return;

    }


    currentCaseIndex =
        caseIndex;

    currentCase =
        caseData;

    currentImages =
        images;

    currentImageIndex =
        imageIndex;


    updateMainImage();

    /*
       Rebuilds the thumbnail list (which also re-applies the
       "selected" highlight) rather than just re-highlighting
       the existing DOM, since this can be reached from a
       different case than the one the thumbnails were last
       built for.
    */
    buildThumbnails();

    buildPanelSelector();

    updateCaseSelection();

}


function selectImage(index) {

    selectImageByID(
        currentCaseIndex,
        index
    );

}


function updateMainImage() {

    const img =
        document.getElementById(
            "main-image"
        );


    if (
        !img ||
        !currentImages.length
    ) {

        return;

    }


    const image =
        currentImages[currentImageIndex];


    if (!image) {

        return;

    }


    img.src =
        image.url;


    img.alt =
        image.type;


    img.title =
        image.type;

}


function updateThumbnailSelection() {

    document
        .querySelectorAll(
            ".thumbnail"
        )
        .forEach(
            thumb => {

                thumb.classList.remove(
                    "selected"
                );

            }
        );


    const currentID =
        getImageID(
            currentCaseIndex,
            currentImageIndex
        );

    const selected =
        document.querySelector(
            `.thumbnail[data-image-id="${currentID}"]`
        );


    if (selected) {

        selected.classList.add(
            "selected"
        );

    }

}


/* ============================================================
   Case Highlight
   ============================================================ */

function updateCaseSelection() {

    const select =
        document.getElementById(
            "case-select"
        );


    if (!select) {

        return;

    }


    select.value =
        currentCaseIndex;

}


/* ============================================================
   Tabs
   ============================================================ */

function setupTabs() {

    document
        .querySelectorAll(
            ".tab-button"
        )
        .forEach(
            button => {

                button.addEventListener(
                    "click",
                    () => {

                        switchTab(
                            button.dataset.tab
                        );

                    }
                );

            }
        );

}


function switchTab(tabID) {

    document
        .querySelectorAll(
            ".tab-panel"
        )
        .forEach(
            panel => {

                panel.classList.remove(
                    "active"
                );

            }
        );


    document
        .querySelectorAll(
            ".tab-button"
        )
        .forEach(
            button => {

                button.classList.remove(
                    "active"
                );

            }
        );


    const panel =
        document.getElementById(
            tabID
        );


    if (panel) {

        panel.classList.add(
            "active"
        );

    }


    const button =
        document.querySelector(
            `[data-tab="${tabID}"]`
        );


    if (button) {

        button.classList.add(
            "active"
        );

    }


    if (
        tabID === "mosaic-tab"
    ) {

        buildMosaicSelector();

        buildMosaic();

    }


    if (
        tabID === "slider-tab"
    ) {

        buildSlider();

    }

}


/* ============================================================
   Interactive Plots (embedded Dropsonde / Nd-LWC pages)
   ============================================================ */

/*
   These are the standalone D3 pages (dropsonde_{campaign}.html,
   nd_lwc_pdf_{campaign}.html - one pair per campaign) written by
   create_website() alongside the dashboard, embedded here via
   <iframe> rather than merged into this page's DOM/JS.

   The campaign list isn't known statically - it's read off of
   whichever campaigns actually show up in the loaded manifest -
   so the sub-tabs/iframes are built dynamically here rather than
   hardcoded in dashboard.html.
*/
function buildInteractiveTabs() {

    const subtabsContainer =
        document.getElementById(
            "interactive-subtabs"
        );

    const framesContainer =
        document.getElementById(
            "interactive-frames"
        );


    if (!subtabsContainer || !framesContainer) {

        return;

    }


    subtabsContainer.innerHTML = "";
    framesContainer.innerHTML = "";


    const campaigns =
        [...new Set(
            dashboardData
                .map(caseData => caseData.campaign)
                .filter(Boolean)
        )];


    const pages = [
        { key: "dropsonde",   label: "Dropsonde",           file: "dropsonde" },
        { key: "ndlwc",       label: "Nd-LWC PDF",          file: "nd_lwc_pdf" },
        { key: "sensitivity", label: "Sensitivity vs Sigma W", file: "sensitivity_vs_sigmaw" },
    ];


    let isFirst = true;

    campaigns.forEach(
        campaign => {

            /*
               Only the pages whose data this run wrote: the manifest
               lists them per entry under "interactive". An older
               manifest without that field shows all three, as before.
            */
            const available =
                new Set(
                    dashboardData
                        .filter(caseData => caseData.campaign === campaign)
                        .flatMap(caseData => caseData.interactive ?? pages.map(p => p.file))
                );

            pages
                .filter(page => available.has(page.file))
                .forEach(
                page => {

                    const frameID =
                        `${page.key}-frame-${campaign}`;


                    const button =
                        document.createElement(
                            "button"
                        );


                    button.className =
                        isFirst
                            ? "tab-button active"
                            : "tab-button";


                    button.dataset.interactiveTab =
                        frameID;


                    button.textContent =
                        `${campaign} — ${page.label}`;


                    button.addEventListener(
                        "click",
                        () => {

                            switchInteractiveFrame(
                                frameID
                            );

                        }
                    );


                    subtabsContainer.appendChild(
                        button
                    );


                    const frame =
                        document.createElement(
                            "iframe"
                        );


                    frame.id =
                        frameID;


                    frame.className =
                        isFirst
                            ? "interactive-frame active"
                            : "interactive-frame";


                    frame.title =
                        `Interactive ${page.label} Plot (${campaign})`;


                    const src =
                        `${page.file}_${campaign}.html`;


                    /*
                       Lazy-load every frame except the
                       first (which is shown by default),
                       so we don't pay to load every
                       campaign's D3 pages up front.
                    */

                    if (isFirst) {

                        frame.src = src;

                    }

                    else {

                        frame.dataset.src = src;

                    }


                    framesContainer.appendChild(
                        frame
                    );


                    isFirst = false;

                }
            );

        }
    );

}


function switchInteractiveFrame(frameID) {

    document
        .querySelectorAll(
            ".interactive-frame"
        )
        .forEach(
            frame => {

                frame.classList.remove(
                    "active"
                );

            }
        );


    document
        .querySelectorAll(
            "[data-interactive-tab]"
        )
        .forEach(
            button => {

                button.classList.remove(
                    "active"
                );

            }
        );


    const frame =
        document.getElementById(
            frameID
        );


    if (frame) {

        frame.classList.add(
            "active"
        );


        /*
           Lazy-load: only point the iframe at its
           page the first time it's shown, so we
           don't pay to load both D3 pages up front.
        */

        if (
            !frame.src &&
            frame.dataset.src
        ) {

            frame.src =
                frame.dataset.src;

        }

    }


    const button =
        document.querySelector(
            `[data-interactive-tab="${frameID}"]`
        );


    if (button) {

        button.classList.add(
            "active"
        );

    }

}


/* ============================================================
   Mosaic Selector
   ============================================================ */

function buildMosaicSelector() {

    const container =
        document.getElementById(
            "mosaic-image-selector"
        );


    if (!container) {

        return;

    }


    container.innerHTML = "";


    const selected =
        Array.from(
            mosaicSelection
        );


    if (!selected.length) {

        const message =
            document.createElement(
                "div"
            );


        message.className =
            "empty-message";


        message.textContent =
            "No images have been selected for the Mosaic. Use the Mosaic checkboxes in the sidebar.";


        container.appendChild(
            message
        );


        return;

    }


    selected.forEach(
        imageID => {

            const item =
                getImageFromID(
                    imageID
                );


            if (!item) {

                return;

            }


            const label =
                document.createElement(
                    "label"
                );


            label.className =
                "mosaic-image-option";


            const checkbox =
                document.createElement(
                    "input"
                );


            checkbox.type =
                "checkbox";


            checkbox.checked = true;


            checkbox.dataset.imageID =
                imageID;


            checkbox.addEventListener(
                "change",
                () => {

                    buildMosaic();

                }
            );


            label.appendChild(
                checkbox
            );


            const text =
                document.createElement(
                    "span"
                );


            text.innerHTML =
                `<strong>${escapeHTML(item.caseData.case)}</strong><br>${escapeHTML(item.image.type)}`;


            label.appendChild(
                text
            );


            container.appendChild(
                label
            );

        }
    );

}


/*
   Get the individual images currently
   visible in the Mosaic.
*/
function getVisibleMosaicImages() {

    const checkboxes =
        document.querySelectorAll(
            "#mosaic-image-selector input[type='checkbox']"
        );


    const visible = [];


    checkboxes.forEach(
        checkbox => {

            if (
                checkbox.checked
            ) {

                visible.push(
                    checkbox.dataset.imageID
                );

            }

        }
    );


    return visible;

}


/* ============================================================
   Mosaic
   ============================================================ */

function buildMosaic() {

    const grid =
        document.getElementById(
            "mosaic-grid"
        );


    if (!grid) {

        return;

    }


    grid.innerHTML = "";


    const visibleIDs =
        getVisibleMosaicImages();


    if (!visibleIDs.length) {

        const message =
            document.createElement(
                "div"
            );


        message.className =
            "empty-message";


        message.textContent =
            mosaicSelection.size
                ? "All selected Mosaic images are currently hidden."
                : "Select images using the Mosaic checkboxes in the sidebar.";


        grid.appendChild(
            message
        );


        return;

    }


    /*
       Each selected image is its own
       independent Mosaic box.
    */

    grid.style.gridTemplateColumns =
        `repeat(auto-fit, minmax(300px, 1fr))`;


    visibleIDs.forEach(
        imageID => {

            const item =
                getImageFromID(
                    imageID
                );


            if (!item) {

                return;

            }


            const panel =
                document.createElement(
                    "div"
                );


            panel.className =
                "mosaic-item";


            /*
               Header
            */

            const header =
                document.createElement(
                    "div"
                );


            header.className =
                "mosaic-item-header";


            const caseTitle =
                document.createElement(
                    "h3"
                );


            caseTitle.textContent =
                item.caseData.case;


            header.appendChild(
                caseTitle
            );


            const imageTitle =
                document.createElement(
                    "div"
                );


            imageTitle.className =
                "mosaic-image-title";


            imageTitle.textContent =
                item.image.type;


            header.appendChild(
                imageTitle
            );


            panel.appendChild(
                header
            );


            /*
               Category
            */

            const category =
                document.createElement(
                    "p"
                );


            category.className =
                "mosaic-category";


            category.textContent =
                item.image.category;


            panel.appendChild(
                category
            );


            /*
               Image
            */

            const img =
                document.createElement(
                    "img"
                );


            img.src =
                item.image.url;


            img.alt =
                `${item.caseData.case} - ${item.image.type}`;


            /*
               Deliberately NOT lazy-loaded: mosaic-grid
               scrolls, and a lazy <img> below the fold
               never fires its load event until scrolled
               into view, which makes saveMosaicPNG's
               html2canvas capture hang indefinitely
               waiting on it.
            */


            panel.appendChild(
                img
            );


            grid.appendChild(
                panel
            );

        }
    );


}


/* ============================================================
   Slider
   ============================================================ */

/*function getSliderImages() {

    return Array.from(
        sliderSelection
    )
    .map(
        imageID =>
            getImageFromID(imageID)
    )
    .filter(
        item =>
            item !== null
    );

}*/

function getSliderPlotTypes() {

    return Array.from(
        sliderPlotTypes
    );

}


function getSliderImages() {

    const images = [];


    const groupKey =
        getSliderPlotTypes()[0];


    if (!groupKey) {

        return images;

    }


    /*
       Walk sliderCaseOrder (the user-controlled,
       drag-and-drop-reorderable order) instead of
       dashboardData, so the Slider plays back in the
       order the user picked.
    */

    sliderCaseOrder.forEach(
        key => {

            if (
                !sliderCaseSelection.has(
                    key
                )
            ) {

                return;

            }


            const caseIndex =
                Number(key);


            const caseData =
                dashboardData[caseIndex];


            if (!caseData) {

                return;

            }


            const imageIndex =
                caseData.urls?.findIndex(
                    image =>
                        getPanelGroupKey(image) ===
                        groupKey
                );


            if (
                imageIndex === undefined ||
                imageIndex < 0
            ) {

                return;

            }


            images.push({

                id:
                    getImageID(
                        caseIndex,
                        imageIndex
                    ),

                caseIndex:
                    caseIndex,

                imageIndex:
                    imageIndex,

                caseData:
                    caseData,

                image:
                    caseData.urls[imageIndex]

            });

        }
    );


    return images;

}


function buildSlider() {

    const container =
        document.getElementById(
            "slider-container"
        );


    if (!container) {

        return;

    }


    /*
       A full rebuild replaces the Play button and the
       image list it was iterating over, so any running
       playback must stop first.
    */

    stopSliderPlayback();


    container.innerHTML = "";


    const plotTypes =
        getSliderPlotTypes();


    /*
       Nothing selected yet.
    */

    if (!plotTypes.length) {

        const message =
            document.createElement(
                "div"
            );


        message.className =
            "empty-message";


        message.textContent =
            "Select a plot type using the Slider checkbox in the sidebar.";


        container.appendChild(
            message
        );


        return;

    }


    /*
       The Slider currently uses one
       plot type (and, if applicable, panel).
    */

    const groupKey =
        plotTypes[0];


    /*
       Find any image matching this group, purely to get a
       human-readable label (type, plus panel label when
       there is one) for the header below - groupKey itself
       is a machine key like "Dropsonde Comparison::open_cell_bias".
    */

    let sampleImage = null;

    for (const caseData of dashboardData) {

        sampleImage =
            (caseData.urls || []).find(
                image =>
                    getPanelGroupKey(image) ===
                    groupKey
            );


        if (sampleImage) break;

    }

    const groupLabel =
        sampleImage
            ? (
                sampleImage.panel
                    ? `${sampleImage.type} — ${sampleImage.panel_label || sampleImage.panel}`
                    : sampleImage.type
            )
            : groupKey;


    /*
       ----------------------------------------------------
       Plot Type Header
       ----------------------------------------------------
    */

    const title =
        document.createElement(
            "h2"
        );


    title.textContent =
        `Slider: ${groupLabel}`;


    container.appendChild(
        title
    );


    /*
       ----------------------------------------------------
       Case Selector
       ----------------------------------------------------
    */

    const caseSelector =
        document.createElement(
            "div"
        );


    caseSelector.className =
        "slider-case-selector";


    const caseHeading =
        document.createElement(
            "h3"
        );


    caseHeading.textContent =
        "Cases";


    caseSelector.appendChild(
        caseHeading
    );


    const caseList =
        document.createElement(
            "div"
        );


    caseList.className =
        "slider-case-list";


    /*
       Cases that contain the selected group (type + panel,
       when there is one), in dashboardData order.
    */

    const eligibleCaseKeys =
        dashboardData
            .map(
                (caseData, caseIndex) => {

                    const hasGroup =
                        caseData.urls?.some(
                            image =>
                                getPanelGroupKey(image) ===
                                groupKey
                        );


                    return hasGroup
                        ? String(caseIndex)
                        : null;

                }
            )
            .filter(
                key => key !== null
            );


    /*
       Keep sliderCaseOrder in sync with the
       eligible cases: drop stale entries, and
       append any that are missing (e.g. the plot
       type checkbox was toggled off then back on).
    */

    sliderCaseOrder =
        sliderCaseOrder.filter(
            key =>
                eligibleCaseKeys.includes(
                    key
                )
        );


    eligibleCaseKeys.forEach(
        key => {

            if (
                !sliderCaseOrder.includes(
                    key
                )
            ) {

                sliderCaseOrder.push(
                    key
                );

            }

        }
    );


    sliderCaseOrder.forEach(
        key => {

            const caseIndex =
                Number(key);


            const caseData =
                dashboardData[caseIndex];


            if (!caseData) {

                return;

            }


            const option =
                document.createElement(
                    "label"
                );


            option.className =
                "slider-case-option";


            option.draggable = true;


            option.dataset.caseKey =
                key;


            /*
               Drag-and-drop reordering.
            */

            option.addEventListener(
                "dragstart",
                () => {

                    draggedSliderCaseKey =
                        key;


                    option.classList.add(
                        "dragging"
                    );

                }
            );


            option.addEventListener(
                "dragend",
                () => {

                    option.classList.remove(
                        "dragging"
                    );

                    draggedSliderCaseKey = null;

                }
            );


            option.addEventListener(
                "dragover",
                event => {

                    event.preventDefault();

                    option.classList.add(
                        "drag-over"
                    );

                }
            );


            option.addEventListener(
                "dragleave",
                () => {

                    option.classList.remove(
                        "drag-over"
                    );

                }
            );


            option.addEventListener(
                "drop",
                event => {

                    event.preventDefault();

                    option.classList.remove(
                        "drag-over"
                    );


                    if (
                        draggedSliderCaseKey === null ||
                        draggedSliderCaseKey === key
                    ) {

                        return;

                    }


                    reorderSliderCases(
                        draggedSliderCaseKey,
                        key
                    );

                }
            );


            const handle =
                document.createElement(
                    "span"
                );


            handle.className =
                "drag-handle";


            handle.textContent =
                "☰";


            handle.title =
                "Drag to reorder";


            option.appendChild(
                handle
            );


            const checkbox =
                document.createElement(
                    "input"
                );


            checkbox.type =
                "checkbox";


            checkbox.checked =
                sliderCaseSelection.has(
                    key
                );


            checkbox.addEventListener(
                "change",
                () => {

                    if (
                        checkbox.checked
                    ) {

                        sliderCaseSelection.add(
                            key
                        );

                    }

                    else {

                        sliderCaseSelection.delete(
                            key
                        );

                    }


                    /*
                       Reset the current position
                       because the number/order of
                       Slider images may have changed.
                    */

                    currentSliderSelectionIndex =
                        0;


                    stopSliderPlayback();


                    buildSlider();

                }
            );


            option.appendChild(
                checkbox
            );


            const text =
                document.createElement(
                    "span"
                );


            text.textContent =
                caseData.case;


            option.appendChild(
                text
            );


            caseList.appendChild(
                option
            );

        }
    );


    caseSelector.appendChild(
        caseList
    );


    container.appendChild(
        caseSelector
    );


    /*
       ----------------------------------------------------
       Selected Slider Images
       ----------------------------------------------------
    */

    const selected =
        getSliderImages();


    if (!selected.length) {

        const message =
            document.createElement(
                "div"
            );


        message.className =
            "empty-message";


        message.textContent =
            "No cases are currently selected.";


        container.appendChild(
            message
        );


        return;

    }


    /*
       Keep index valid.
    */

    if (
        currentSliderSelectionIndex >=
        selected.length
    ) {

        currentSliderSelectionIndex =
            0;

    }


    /*
       ----------------------------------------------------
       Slider Controls
       ----------------------------------------------------
    */

    const controls =
        document.createElement(
            "div"
        );


    controls.className =
        "slider-controls";


    /*
       Current image/case indicator
    */

    const indicator =
        document.createElement(
            "span"
        );


    indicator.id =
        "slider-indicator";


    indicator.className =
        "slider-indicator";


    const current =
        selected[
            currentSliderSelectionIndex
        ];


    indicator.textContent =
        `${current.caseData.case} — ${current.image.type} (${currentSliderSelectionIndex + 1}/${selected.length})`;


    controls.appendChild(
        indicator
    );


    /*
       Previous
    */

    const previous =
        document.createElement(
            "button"
        );


    previous.className =
        "control-button";


    previous.textContent =
        "◀ Previous";


    previous.onclick =
        () => {

            stopSliderPlayback();

            updateSliderPosition(-1);

        };


    controls.appendChild(
        previous
    );


    /*
       Next
    */

    const next =
        document.createElement(
            "button"
        );


    next.className =
        "control-button";


    next.textContent =
        "Next ▶";


    next.onclick =
        () => {

            stopSliderPlayback();

            updateSliderPosition(1);

        };


    controls.appendChild(
        next
    );


    /*
       Play / Pause
    */

    const playButton =
        document.createElement(
            "button"
        );


    playButton.id =
        "slider-play-button";


    playButton.className =
        "control-button";


    playButton.textContent =
        sliderPlaybackTimer
            ? "⏸ Pause"
            : "▶ Play";


    playButton.disabled =
        selected.length < 2;


    playButton.onclick =
        toggleSliderPlayback;


    controls.appendChild(
        playButton
    );


    /*
       Save GIF
    */

    const gifButton =
        document.createElement(
            "button"
        );


    gifButton.id =
        "slider-gif-button";


    gifButton.className =
        "control-button";


    gifButton.textContent =
        "Save GIF";


    gifButton.onclick =
        saveSliderGIF;


    controls.appendChild(
        gifButton
    );


    container.appendChild(
        controls
    );


    /*
       ----------------------------------------------------
       Viewer
       ----------------------------------------------------
    */

    const viewer =
        document.createElement(
            "div"
        );


    viewer.id =
        "slider-viewer";


    viewer.className =
        "slider-viewer";


    container.appendChild(
        viewer
    );


    renderSlider();

}


function renderSlider() {

    const selected =
        getSliderImages();


    if (!selected.length) {

        return;

    }


    const item =
        selected[
            currentSliderSelectionIndex
        ];


    const viewer =
        document.getElementById(
            "slider-viewer"
        );


    if (!viewer) {

        return;

    }


    viewer.innerHTML = "";


    const img =
        document.createElement(
            "img"
        );


    img.src =
        item.image.url;


    img.alt =
        `${item.caseData.case} - ${item.image.type}`;


    viewer.appendChild(
        img
    );


}


/* ============================================================
   Slider Navigation / Playback

   Moves through the Slider images WITHOUT rebuilding the
   whole tab, so Play/Pause and Previous/Next stay smooth
   and don't fight over the setInterval handle.
   ============================================================ */

function updateSliderPosition(delta) {

    const selected =
        getSliderImages();


    if (!selected.length) {

        return;

    }


    currentSliderSelectionIndex =
        (
            currentSliderSelectionIndex +
            delta +
            selected.length
        ) % selected.length;


    renderSlider();


    const indicator =
        document.getElementById(
            "slider-indicator"
        );


    if (indicator) {

        const current =
            selected[
                currentSliderSelectionIndex
            ];


        indicator.textContent =
            `${current.caseData.case} — ${current.image.type} (${currentSliderSelectionIndex + 1}/${selected.length})`;

    }

}


function toggleSliderPlayback() {

    if (sliderPlaybackTimer) {

        stopSliderPlayback();

        return;

    }


    const selected =
        getSliderImages();


    if (selected.length < 2) {

        return;

    }


    sliderPlaybackTimer =
        setInterval(
            () => {

                updateSliderPosition(1);

            },
            sliderPlaybackIntervalMs
        );


    updatePlayButtonLabel();

}


function stopSliderPlayback() {

    if (sliderPlaybackTimer) {

        clearInterval(
            sliderPlaybackTimer
        );


        sliderPlaybackTimer = null;

    }


    updatePlayButtonLabel();

}


function updatePlayButtonLabel() {

    const button =
        document.getElementById(
            "slider-play-button"
        );


    if (!button) {

        return;

    }


    button.textContent =
        sliderPlaybackTimer
            ? "⏸ Pause"
            : "▶ Play";


    button.classList.toggle(
        "active",
        Boolean(
            sliderPlaybackTimer
        )
    );

}


/* ============================================================
   Slider Case Reordering (drag-and-drop)
   ============================================================ */

function reorderSliderCases(
    sourceKey,
    targetKey
) {

    const fromIndex =
        sliderCaseOrder.indexOf(
            sourceKey
        );


    const toIndex =
        sliderCaseOrder.indexOf(
            targetKey
        );


    if (
        fromIndex < 0 ||
        toIndex < 0
    ) {

        return;

    }


    sliderCaseOrder.splice(
        fromIndex,
        1
    );


    sliderCaseOrder.splice(
        toIndex,
        0,
        sourceKey
    );


    currentSliderSelectionIndex = 0;


    buildSlider();

}


/* ============================================================
   Save Mosaic
   ============================================================ */

async function saveMosaicPNG() {

    const grid =
        document.getElementById(
            "mosaic-grid"
        );


    if (
        !grid ||
        !grid.children.length
    ) {

        alert(
            "There is no Mosaic to save."
        );

        return;

    }


    if (
        typeof html2canvas ===
        "undefined"
    ) {

        alert(
            "html2canvas is not loaded."
        );

        return;

    }


    const button =
        document.getElementById(
            "save-mosaic"
        );


    const originalLabel =
        button?.textContent;


    if (button) {

        button.disabled = true;

        button.textContent =
            "Saving…";

    }


    try {

        /*
           Make sure every tile has actually finished
           loading before handing the grid to html2canvas.
           An incomplete <img> can otherwise make the
           capture hang waiting on a load event that
           never fires.
        */

        await waitForImagesToLoad(
            grid,
            8000
        );


        const canvas =
            await withTimeout(
                html2canvas(
                    grid,
                    {
                        backgroundColor:
                            "#121212",

                        scale: 2,

                        /*
                           Allows images served from a
                           different origin (e.g. a data
                           server) to still be captured.
                        */
                        useCORS: true,

                        /*
                           Bounds how long html2canvas will
                           wait on any single image before
                           giving up on it, so a bad/slow
                           image can't hang the save forever.
                        */
                        imageTimeout: 8000
                    }
                ),
                20000,
                "Timed out capturing the Mosaic."
            );


        const link =
            document.createElement(
                "a"
            );


        link.download =
            "CESM_Mosaic.png";


        link.href =
            canvas.toDataURL(
                "image/png"
            );


        link.click();

    }

    catch(error) {

        console.error(
            "Could not save Mosaic:",
            error
        );


        alert(
            "Could not save Mosaic."
        );

    }

    finally {

        if (button) {

            button.disabled = false;

            button.textContent =
                originalLabel;

        }

    }

}


/* ============================================================
   gif.js Worker Script

   Cached so it's only downloaded once per page load.
   ============================================================ */

let gifWorkerScriptURLPromise = null;


function getGifWorkerScriptURL() {

    if (!gifWorkerScriptURLPromise) {

        gifWorkerScriptURLPromise =
            fetch(
                "https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.worker.js"
            )
            .then(
                response => {

                    if (!response.ok) {

                        throw new Error(
                            "Could not download gif.js worker script."
                        );

                    }


                    return response.blob();

                }
            )
            .then(
                blob =>
                    URL.createObjectURL(
                        blob
                    )
            );

    }


    return gifWorkerScriptURLPromise;

}


/* ============================================================
   Save Slider GIF
   ============================================================ */

async function saveSliderGIF() {

    if (
        typeof GIF ===
        "undefined"
    ) {

        alert(
            "gif.js is not loaded."
        );

        return;

    }


    const selected =
        getSliderImages();


    if (!selected.length) {

        alert(
            "No images are selected for the Slider."
        );

        return;

    }


    stopSliderPlayback();


    const button =
        document.getElementById(
            "slider-gif-button"
        );


    const originalLabel =
        button?.textContent;


    if (button) {

        button.disabled = true;

        button.textContent =
            "Rendering GIF…";

    }


    /*
       gif.js spins up real Worker threads, and a dedicated
       Worker's script must be same-origin — browsers throw
       a SecurityError (uncaught, since this happens inside
       gif.js) for a raw cdnjs URL. Downloading it ourselves
       and handing gif.js a blob: URL sidesteps that; this is
       gif.js's documented workaround for CDN-hosted builds.
    */

    let workerScriptURL;


    try {

        workerScriptURL =
            await getGifWorkerScriptURL();

    }

    catch(error) {

        console.error(
            "Could not prepare gif.js worker:",
            error
        );


        /*
           Let a later click retry the download instead of
           being stuck on this one failed attempt forever.
        */
        gifWorkerScriptURLPromise = null;


        alert(
            "Could not save GIF: the GIF encoder failed to load."
        );


        if (button) {

            button.disabled = false;

            button.textContent =
                originalLabel;

        }


        return;

    }


    /*
       The GIF contains the selected individual
       Slider images in their current order.
    */

    const gif =
        new GIF({

            workers: 2,

            quality: 10,

            width: 1200,

            height: 800,

            workerScript:
                workerScriptURL

        });


    let framesAdded = 0;


    for (
        const item of selected
    ) {

        try {

            const loaded =
                await loadImage(
                    item.image.url
                );


            const canvas =
                document.createElement(
                    "canvas"
                );


            canvas.width =
                1200;


            canvas.height =
                800;


            const ctx =
                canvas.getContext(
                    "2d"
                );


            ctx.fillStyle =
                "#121212";


            ctx.fillRect(
                0,
                0,
                1200,
                800
            );


            const scale =
                Math.min(
                    1200 /
                        loaded.width,

                    800 /
                        loaded.height
                );


            const width =
                loaded.width *
                scale;


            const height =
                loaded.height *
                scale;


            const x =
                (1200 - width) / 2;


            const y =
                (800 - height) / 2;


            ctx.drawImage(
                loaded,
                x,
                y,
                width,
                height
            );


            /*
               Case + image name
            */

            ctx.font =
                "bold 24px Arial";


            ctx.fillStyle =
                "white";


            ctx.fillText(
                `${item.caseData.case} — ${item.image.type}`,
                25,
                40
            );


            gif.addFrame(
                canvas,
                {
                    delay: 1500
                }
            );


            framesAdded++;

        }

        catch(error) {

            console.error(
                "Could not load:",
                item.image.url,
                error
            );

        }

    }


    if (!framesAdded) {

        alert(
            "Could not save GIF: none of the Slider images could be loaded."
        );


        if (button) {

            button.disabled = false;

            button.textContent =
                originalLabel;

        }


        return;

    }


    /*
       Guards against restoring/alerting twice if the safety
       timeout below fires around the same moment "finished"
       does.
    */
    let settled = false;


    const restoreButton = () => {

        if (button) {

            button.disabled = false;

            button.textContent =
                originalLabel;

        }

    };


    /*
       gif.js has no built-in failure event for a worker
       that silently dies, so without this, an unexpected
       worker error leaves the button stuck on "Rendering
       GIF…" forever with no way out.
    */
    const safetyTimer =
        setTimeout(
            () => {

                if (settled) {

                    return;

                }


                settled = true;


                console.error(
                    "Saving the Slider GIF timed out."
                );


                alert(
                    "Could not save GIF: rendering timed out."
                );


                restoreButton();

            },
            60000
        );


    gif.on(
        "progress",
        ratio => {

            if (
                !settled &&
                button
            ) {

                button.textContent =
                    `Rendering GIF… ${Math.round(ratio * 100)}%`;

            }

        }
    );


    gif.on(
        "finished",
        blob => {

            if (settled) {

                return;

            }


            settled = true;

            clearTimeout(
                safetyTimer
            );


            const url =
                URL.createObjectURL(
                    blob
                );


            const link =
                document.createElement(
                    "a"
                );


            link.href =
                url;


            link.download =
                "CESM_Slider.gif";


            link.click();


            setTimeout(
                () => {

                    URL.revokeObjectURL(
                        url
                    );

                },
                1000
            );


            restoreButton();

        }
    );


    try {

        gif.render();

    }

    catch(error) {

        if (!settled) {

            settled = true;

            clearTimeout(
                safetyTimer
            );


            console.error(
                "Could not render GIF:",
                error
            );


            alert(
                "Could not save GIF."
            );


            restoreButton();

        }

    }

}


/* ============================================================
   Async Helpers

   Both saveMosaicPNG and saveSliderGIF depend on external
   image loads / libraries that can, in the wrong conditions,
   never resolve on their own. These bound how long we'll
   ever wait so a save action can't hang the UI forever.
   ============================================================ */

function waitForImagesToLoad(
    container,
    timeoutMs
) {

    const pending =
        Array.from(
            container.querySelectorAll(
                "img"
            )
        )
        .filter(
            img => !img.complete
        );


    if (!pending.length) {

        return Promise.resolve();

    }


    return Promise.race([

        Promise.all(
            pending.map(
                img =>
                    new Promise(
                        resolve => {

                            img.addEventListener(
                                "load",
                                resolve,
                                { once: true }
                            );


                            img.addEventListener(
                                "error",
                                resolve,
                                { once: true }
                            );

                        }
                    )
            )
        ),

        new Promise(
            resolve =>
                setTimeout(
                    resolve,
                    timeoutMs
                )
        )

    ]);

}


function withTimeout(
    promise,
    ms,
    message
) {

    return new Promise(
        (resolve, reject) => {

            const timer =
                setTimeout(
                    () => reject(
                        new Error(
                            message
                        )
                    ),
                    ms
                );


            promise.then(
                value => {

                    clearTimeout(
                        timer
                    );

                    resolve(
                        value
                    );

                },
                error => {

                    clearTimeout(
                        timer
                    );

                    reject(
                        error
                    );

                }
            );

        }
    );

}


/* ============================================================
   Image Loader
   ============================================================ */

function loadImage(src) {

    return new Promise(
        (resolve, reject) => {

            const img =
                new Image();


            /*
               Avoids tainting the canvas (and breaking
               gif.js frame capture) when images are
               served from a different origin.
            */
            img.crossOrigin =
                "anonymous";


            img.onload =
                () => resolve(img);


            img.onerror =
                () => reject(
                    new Error(
                        `Could not load image: ${src}`
                    )
                );


            img.src =
                src;

        }
    );

}


/* ============================================================
   HTML Escaping
   ============================================================ */

function escapeHTML(value) {

    return String(value)
        .replace(
            /&/g,
            "&amp;"
        )
        .replace(
            /</g,
            "&lt;"
        )
        .replace(
            />/g,
            "&gt;"
        )
        .replace(
            /"/g,
            "&quot;"
        )
        .replace(
            /'/g,
            "&#039;"
        );

}