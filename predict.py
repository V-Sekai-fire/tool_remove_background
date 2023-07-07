from typing import Optional
import uuid6

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.session = new_session("u2net")

    # Define the arguments and types the model takes as input
    def predict(self, 
                image: Optional[Path] = Input(description="Image file"), 
                image_url: Optional[str] = Input(description="Input image URL"),
                only_mask: bool = Input(default=False, description="Only return the mask"),
                alpha_matting: bool = Input(default=False, description="Use alpha matting")) -> Path:
        """Run a single prediction on the model"""
        
        # Load the image from file or URL
        if image is not None:
            img = Image.open(image)
        elif image_url is not None:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
        else:
            raise ValueError("Either 'image' or 'image_url' must be provided.")
        
        # Remove the background
        output = remove(np.array(img), session=self.session, alpha_matting=alpha_matting)
        
        if only_mask:
            # If only_mask is True, return the mask instead of the image with background removed
            mask = output[..., 3]
            output_image = Image.fromarray(mask)
        else:
            output_image = Image.fromarray(output)
        
        # Save the output image
        out_path = Path(tempfile.mkdtemp()) / f"{uuid6.uuid7()}.png"
        output_image.save(str(out_path))
        return out_path
