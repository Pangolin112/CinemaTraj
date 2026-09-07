from extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer

def draw_json(c2ws, vis_path):

    output_dir = os.path.dirname(vis_path)
    parent_dir = os.path.dirname(output_dir)

    rangesize = torch.max(torch.abs(torch.tensor(c2ws[:, :3, 3]))) * 1.1

    # Prepare visualizer
    visualizer = CameraPoseVisualizer([-rangesize, rangesize], [-rangesize, rangesize], [-rangesize, rangesize])

    num_matrices = c2ws.shape[0]

    # Create a color gradient from red to purple
    colors = plt.cm.rainbow(np.linspace(1, 0, num_matrices))

    # Create three views
    views = [
        {'elev': 90, 'azim': -90, 'name': 'front'},
        {'elev': 180, 'azim': -90, 'name': 'top'},
        {'elev': 0, 'azim': 0, 'name': 'side'}
    ]
    
    image_paths = []

    for view in views:
        fig = plt.figure(figsize=(12, 12))  # Each image will be 4x12 inches
        visualizer = CameraPoseVisualizer([-rangesize, rangesize], [-rangesize, rangesize], [-rangesize, rangesize])

        for i in range(num_matrices):
            color = colors[i]
            # print(c2ws[i])
            visualizer.extrinsic2pyramid(c2ws[i], color, rangesize / 4)
        
        visualizer.ax.view_init(elev=view['elev'], azim=view['azim'])
        
        # Save each view as a separate image
        image_path = f"{parent_dir}/{view['name']}_view.png"
        os.makedirs(output_dir, exist_ok=True)
        visualizer.save(image_path)
        image_paths.append(image_path)
    
        
    # Now combine the three images horizontally
    images = [Image.open(img_path) for img_path in image_paths]
    images[-1] = images[-1].rotate(90, expand=True)

    # Resize images to fit the desired final size
    images = [img.crop((420, 420, 1980, 1980)) for img in images]
    images_resized = [img.resize((341, 341)) for img in images]

    # Concatenate images horizontally
    combined_image = np.concatenate([np.array(img) for img in images_resized], axis=1)

    # Save the final combined image
    final_image = Image.fromarray(combined_image)
    final_image.save(vis_path)

    print(f"Combined image saved at {vis_path}")