document.addEventListener('keydown', function (e) {
    if (e.shiftKey && e.key === 'Enter') {
        const nextButton = document.getElementById('next-trigger-button');
        if (nextButton) {
            nextButton.click();
        }
    }
});
