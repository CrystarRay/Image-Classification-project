from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .models import classify_image
from .forms import ImageUploadForm


def index(request):
    # If the request method is POST, it means an image is being uploaded.
    if request.method == 'POST':
        # Get the form data from the request.
        form = ImageUploadForm(request.POST, request.FILES)

        # Check if the form data is valid.
        if form.is_valid():
            # Get the image from the form.
            image = form.cleaned_data['image']

            # Save the image temporarily in the file system.
            fs = FileSystemStorage()
            filename = fs.save("temp.jpg", image)
            uploaded_file_url = fs.path(filename)

            # Call the classify_image function to get the prediction.
            predictions = classify_image(uploaded_file_url)
            prediction = predictions[0]
            rounded_predictions =  map(lambda x : int(x * 100), predictions[1])

            # Delete the temporary image file.
            fs.delete(filename)

            # Render the template with the classification message.
            return render(request, 'index.html',
                          {'classification_message': f'The image is classified as number: {prediction}',
                          'classifications': rounded_predictions})

    # If the request method is not POST, display the empty form.
    else:
        form = ImageUploadForm()

    # Render the template with the form.
    return render(request, 'index.html', {'form': form})