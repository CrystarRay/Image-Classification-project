from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .models import classify_image
from .forms import ImageUploadForm

def index(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            fs = FileSystemStorage()
            filename = fs.save("temp.jpg", image)
            uploaded_file_url = fs.path(filename)

            prediction = classify_image(uploaded_file_url)
            fs.delete(filename)

            return render(request, 'index.html', {'form': form, 'prediction': prediction})
    else:
        form = ImageUploadForm()
    return render(request, 'index.html', {'form': form})