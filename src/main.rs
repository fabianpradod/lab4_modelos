extern crate sdl2;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Point;
use sdl2::render::WindowCanvas;
use std::error::Error;
use std::f32::consts::FRAC_PI_4;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

const SCREEN_WIDTH: i32 = 960;
const SCREEN_HEIGHT: i32 = 720;
const OBJ_PATH: &str = "spaceship.obj";
const BASE_TILT_X: f32 = -1.05;
const BASE_ROTATION_Y: f32 = -FRAC_PI_4;

#[derive(Clone, Copy, Debug, Default)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    fn normalize(self) -> Self {
        let len = self.length();
        if len > 1e-6 { self / len } else { self }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl std::ops::Div<f32> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ProjectedVertex {
    x: f32,
    y: f32,
    z: f32,
}

struct Mesh {
    vertices: Vec<Vec3>,
    faces: Vec<[usize; 3]>,
}

impl Mesh {
    fn load_from_obj(path: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        for line_result in reader.lines() {
            let line = line_result?;
            let trimmed = line.trim();

            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let mut parts = trimmed.split_whitespace();
            match parts.next() {
                Some("v") => {
                    let x: f32 = parts.next().unwrap_or("0.0").parse()?;
                    let y: f32 = parts.next().unwrap_or("0.0").parse()?;
                    let z: f32 = parts.next().unwrap_or("0.0").parse()?;
                    vertices.push(Vec3::new(x, y, z));
                }
                Some("f") => {
                    let mut indices = Vec::new();
                    for token in parts {
                        let index_str = token.split('/').next().unwrap_or("");
                        if index_str.is_empty() {
                            continue;
                        }
                        let index: usize = index_str.parse()?;
                        if index == 0 {
                            continue;
                        }
                        indices.push(index - 1);
                    }

                    if indices.len() >= 3 {
                        for i in 1..indices.len() - 1 {
                            faces.push([indices[0], indices[i], indices[i + 1]]);
                        }
                    }
                }
                _ => continue,
            }
        }

        if vertices.is_empty() {
            return Err("OBJ file does not contain vertices".into());
        }

        let mut mesh = Mesh { vertices, faces };
        mesh.normalize_vertices();
        Ok(mesh)
    }

    fn normalize_vertices(&mut self) {
        if self.vertices.is_empty() {
            return;
        }

        let mut min_x = self.vertices[0].x;
        let mut max_x = self.vertices[0].x;
        let mut min_y = self.vertices[0].y;
        let mut max_y = self.vertices[0].y;
        let mut min_z = self.vertices[0].z;
        let mut max_z = self.vertices[0].z;

        for v in &self.vertices {
            min_x = min_x.min(v.x);
            max_x = max_x.max(v.x);
            min_y = min_y.min(v.y);
            max_y = max_y.max(v.y);
            min_z = min_z.min(v.z);
            max_z = max_z.max(v.z);
        }

        let center = Vec3::new(
            (min_x + max_x) * 0.5,
            (min_y + max_y) * 0.5,
            (min_z + max_z) * 0.5,
        );

        let mut max_extent = 0.0f32;
        for v in &self.vertices {
            let offset = *v - center;
            max_extent = max_extent
                .max(offset.x.abs())
                .max(offset.y.abs())
                .max(offset.z.abs());
        }

        if max_extent < 1e-6 {
            max_extent = 1.0;
        }

        for v in &mut self.vertices {
            *v = (*v - center) / max_extent;
        }
    }
}

struct FrameBuffer {
    width: i32,
    height: i32,
    pixels: Vec<Color>,
    depth: Vec<f32>,
}

impl FrameBuffer {
    fn new(width: i32, height: i32) -> Self {
        let pixel_count = (width * height) as usize;
        FrameBuffer {
            width,
            height,
            pixels: vec![Color::RGB(0, 0, 0); pixel_count],
            depth: vec![f32::INFINITY; pixel_count],
        }
    }

    fn clear(&mut self, color: Color) {
        for pixel in &mut self.pixels {
            *pixel = color;
        }
        for depth in &mut self.depth {
            *depth = f32::INFINITY;
        }
    }

    fn try_set_pixel(&mut self, x: i32, y: i32, color: Color, depth: f32) {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return;
        }

        let index = (y * self.width + x) as usize;
        if depth < self.depth[index] {
            self.depth[index] = depth;
            self.pixels[index] = color;
        }
    }

    fn draw_to_canvas(&self, canvas: &mut WindowCanvas) -> Result<(), String> {
        for y in 0..self.height {
            for x in 0..self.width {
                let index = (y * self.width + x) as usize;
                canvas.set_draw_color(self.pixels[index]);
                canvas.draw_point(Point::new(x, y))?;
            }
        }
        Ok(())
    }

    fn save_png(&self, filename: &str) -> Result<(), Box<dyn Error>> {
        use std::io::BufWriter;

        let file = File::create(filename)?;
        let mut writer = BufWriter::new(file);

        let mut encoder = png::Encoder::new(&mut writer, self.width as u32, self.height as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut png_writer = encoder.write_header()?;

        let mut rgb_data = Vec::with_capacity(self.pixels.len() * 3);
        for pixel in &self.pixels {
            rgb_data.push(pixel.r);
            rgb_data.push(pixel.g);
            rgb_data.push(pixel.b);
        }

        png_writer.write_image_data(&rgb_data)?;
        Ok(())
    }
}

fn rotate_y(v: Vec3, angle: f32) -> Vec3 {
    let cos = angle.cos();
    let sin = angle.sin();
    Vec3::new(v.x * cos + v.z * sin, v.y, -v.x * sin + v.z * cos)
}

fn rotate_x(v: Vec3, angle: f32) -> Vec3 {
    let cos = angle.cos();
    let sin = angle.sin();
    Vec3::new(v.x, v.y * cos - v.z * sin, v.y * sin + v.z * cos)
}

fn render_mesh(buffer: &mut FrameBuffer, mesh: &Mesh, rotation: f32) {
    let width = buffer.width as f32;
    let height = buffer.height as f32;
    let aspect = width / height;
    let fov = 60.0_f32.to_radians();
    let focal_length = 1.0 / (fov * 0.5).tan();
    let light_direction = Vec3::new(-0.4, 0.9, -1.0).normalize();
    let camera_offset = Vec3::new(0.0, 0.0, 3.0);

    for face in &mesh.faces {
        let mut world = [Vec3::default(); 3];
        let mut projected = [ProjectedVertex::default(); 3];
        let mut skip_face = false;

        for (i, &index) in face.iter().enumerate() {
            let vertex = mesh.vertices.get(index).copied().unwrap_or_default();
            let rotated = rotate_y(rotate_x(vertex, BASE_TILT_X), rotation + BASE_ROTATION_Y);
            world[i] = rotated;

            let camera_space = rotated + camera_offset;
            if camera_space.z <= 0.1 {
                skip_face = true;
                break;
            }

            let ndc_x = (camera_space.x * focal_length) / (camera_space.z * aspect);
            let ndc_y = (camera_space.y * focal_length) / camera_space.z;

            let screen_x = ((ndc_x + 1.0) * 0.5) * (width - 1.0);
            let screen_y = ((1.0 - ndc_y) * 0.5) * (height - 1.0);

            projected[i] = ProjectedVertex {
                x: screen_x,
                y: screen_y,
                z: camera_space.z,
            };
        }

        if skip_face {
            continue;
        }

        let normal = (world[1] - world[0]).cross(world[2] - world[0]).normalize();

        if normal.z >= 0.0 {
            continue;
        }

        let intensity = normal.dot(light_direction).max(0.0);
        if intensity <= 0.0 {
            continue;
        }

        let color = shade_color(intensity);
        draw_filled_triangle(buffer, projected, color);
    }
}

fn shade_color(intensity: f32) -> Color {
    let ambient = 0.15;
    let final_intensity = (ambient + (1.0 - ambient) * intensity).clamp(0.0, 1.0);

    let r = (40.0 + 180.0 * final_intensity).clamp(0.0, 255.0) as u8;
    let g = (50.0 + 150.0 * final_intensity).clamp(0.0, 255.0) as u8;
    let b = (110.0 + 120.0 * final_intensity).clamp(0.0, 255.0) as u8;

    Color::RGB(r, g, b)
}

fn draw_filled_triangle(buffer: &mut FrameBuffer, vertices: [ProjectedVertex; 3], color: Color) {
    let min_x = vertices
        .iter()
        .map(|v| v.x)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as i32;
    let max_x = vertices
        .iter()
        .map(|v| v.x)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(buffer.width as f32 - 1.0) as i32;
    let min_y = vertices
        .iter()
        .map(|v| v.y)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as i32;
    let max_y = vertices
        .iter()
        .map(|v| v.y)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(buffer.height as f32 - 1.0) as i32;

    if min_x > max_x || min_y > max_y {
        return;
    }

    let area = edge_function(&vertices[0], &vertices[1], vertices[2].x, vertices[2].y);
    if area.abs() < 1e-6 {
        return;
    }
    let inv_area = 1.0 / area;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;

            let w0 = edge_function(&vertices[1], &vertices[2], px, py);
            let w1 = edge_function(&vertices[2], &vertices[0], px, py);
            let w2 = edge_function(&vertices[0], &vertices[1], px, py);

            if (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0) {
                let bary0 = w0 * inv_area;
                let bary1 = w1 * inv_area;
                let bary2 = w2 * inv_area;

                let depth = bary0 * vertices[0].z + bary1 * vertices[1].z + bary2 * vertices[2].z;
                buffer.try_set_pixel(x, y, color, depth);
            }
        }
    }
}

fn edge_function(a: &ProjectedVertex, b: &ProjectedVertex, px: f32, py: f32) -> f32 {
    (px - a.x) * (b.y - a.y) - (py - a.y) * (b.x - a.x)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mesh = Mesh::load_from_obj(OBJ_PATH)?;

    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;
    let window = video_subsystem
        .window(
            "Lab 4 - OBJ Renderer",
            SCREEN_WIDTH as u32,
            SCREEN_HEIGHT as u32,
        )
        .position_centered()
        .build()?;

    let mut canvas = window
        .into_canvas()
        .accelerated()
        .present_vsync()
        .build()
        .map_err(|e| format!("Failed to create canvas: {}", e))?;

    let mut event_pump = sdl_context.event_pump()?;
    let mut framebuffer = FrameBuffer::new(SCREEN_WIDTH, SCREEN_HEIGHT);

    let mut rotation = 0.0f32;
    let mut previous_frame = Instant::now();
    let mut saved_frame = false;

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                _ => {}
            }
        }

        let now = Instant::now();
        let delta = now - previous_frame;
        previous_frame = now;
        rotation += delta.as_secs_f32() * 0.7;

        framebuffer.clear(Color::RGB(12, 12, 22));
        render_mesh(&mut framebuffer, &mesh, rotation);

        if !saved_frame {
            framebuffer.save_png("spaceship_render.png")?;
            saved_frame = true;
        }

        framebuffer.draw_to_canvas(&mut canvas)?;
        canvas.present();

        std::thread::sleep(Duration::from_millis(16));
    }

    Ok(())
}
