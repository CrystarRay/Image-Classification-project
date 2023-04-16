from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import ImageUploadForm
from PIL import Image

# Replace this import with your actual image classification model and function
from .models import classify_image

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = Image.open(request.FILES['image'])

            # Convert the image to RGB if it has an alpha channel (transparency)
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')

            # Save the image as a JPEG
            image.save('media/classification_temp.jpg', 'JPEG', quality=90)

            # Perform image classification
            classification_result = classify_image('media/classification_temp.jpg')

            # Render the result page
            return render(request, 'result.html', {'classification_result': classification_result})
    else:
        form = ImageUploadForm()

    return render(request, 'upload.html', {'form': form})

def result(request):
    return render(request, 'result.html')