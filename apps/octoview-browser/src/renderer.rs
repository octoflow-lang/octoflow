use crate::framebuffer::Framebuffer;
use crate::layout::{LayoutBox, LayoutContent};
use crate::text::TextRenderer;

/// Render laid-out page content onto the CPU framebuffer.
/// `viewport_top` is the Y coordinate where page content starts (below toolbar).
/// `viewport_height` is the available height for content (between toolbar and status bar).
pub fn render_page(
    fb: &mut Framebuffer,
    boxes: &[LayoutBox],
    text_renderer: &mut TextRenderer,
    scroll_y: f32,
    viewport_top: f32,
    viewport_height: f32,
) {
    let viewport_bottom = viewport_top + viewport_height;

    for b in boxes {
        let screen_y = b.y - scroll_y + viewport_top;

        // Cull: skip boxes entirely outside viewport
        if screen_y + b.height < viewport_top || screen_y > viewport_bottom {
            continue;
        }

        let sx = b.x as i32;
        let sy = screen_y as i32;
        let color = b.style.text_color;
        let font_size = b.style.font_size;
        let font_style = b.style.font_style;

        // Draw background if present
        if let Some(bg) = b.style.bg_color {
            fb.draw_rect(sx, sy, b.width as u32, b.height as u32, bg[0], bg[1], bg[2]);
        }

        // Draw border if present
        if b.style.border_width > 0.0 {
            let bc = b.style.border_color.unwrap_or([69, 71, 90]);
            let bw = b.style.border_width as u32;
            let w = b.width as u32;
            let h = b.height as u32;
            // Top
            fb.draw_rect(sx, sy, w, bw.min(h), bc[0], bc[1], bc[2]);
            // Bottom
            if h > bw {
                fb.draw_rect(sx, sy + (h - bw) as i32, w, bw, bc[0], bc[1], bc[2]);
            }
            // Left
            fb.draw_rect(sx, sy, bw.min(w), h, bc[0], bc[1], bc[2]);
            // Right
            if w > bw {
                fb.draw_rect(sx + (w - bw) as i32, sy, bw, h, bc[0], bc[1], bc[2]);
            }
        }

        match &b.content {
            LayoutContent::Text(lines) => {
                for line in lines {
                    let lx = sx + line.x_offset as i32;
                    let ly = sy + line.y_offset as i32;
                    text_renderer.draw_text(
                        fb, &line.text, lx, ly, font_size, font_style,
                        color[0], color[1], color[2],
                    );
                }
            }

            LayoutContent::Link(lines, _href) => {
                let line_h = text_renderer.line_height(font_size);
                for line in lines {
                    let lx = sx + line.x_offset as i32;
                    let ly = sy + line.y_offset as i32;
                    text_renderer.draw_text(
                        fb, &line.text, lx, ly, font_size, font_style,
                        color[0], color[1], color[2],
                    );
                    // Underline
                    if b.style.text_decoration_underline {
                        let (tw, _) = text_renderer.measure_text(&line.text, font_size, font_style);
                        let uy = ly + line_h as i32 - 2;
                        fb.draw_hline(lx, uy, tw as u32, color[0], color[1], color[2]);
                    }
                }
            }

            LayoutContent::CodeBlock(lines) => {
                // Background already drawn above via bg_color
                for line in lines {
                    let lx = sx + line.x_offset as i32;
                    let ly = sy + line.y_offset as i32;
                    text_renderer.draw_text(
                        fb, &line.text, lx, ly, font_size, font_style,
                        color[0], color[1], color[2],
                    );
                }
            }

            LayoutContent::Blockquote(lines) => {
                // Draw left border (4px wide, muted blue)
                fb.draw_rect(sx, sy, 4, b.height as u32, 100, 120, 180);
                for line in lines {
                    let lx = sx + line.x_offset as i32;
                    let ly = sy + line.y_offset as i32;
                    text_renderer.draw_text(
                        fb, &line.text, lx, ly, font_size, font_style,
                        color[0], color[1], color[2],
                    );
                }
            }

            LayoutContent::ListItem(bullet, lines) => {
                // Draw bullet
                let bullet_x = sx + 8;
                let bullet_y = sy;
                text_renderer.draw_text(
                    fb, bullet, bullet_x, bullet_y, font_size, font_style,
                    color[0], color[1], color[2],
                );
                // Draw text lines
                for line in lines {
                    let lx = sx + line.x_offset as i32;
                    let ly = sy + line.y_offset as i32;
                    text_renderer.draw_text(
                        fb, &line.text, lx, ly, font_size, font_style,
                        color[0], color[1], color[2],
                    );
                }
            }

            LayoutContent::Separator => {
                fb.draw_hline(sx, sy, b.width as u32, color[0], color[1], color[2]);
            }

            LayoutContent::Image {
                ref alt,
                ref pixels,
                img_width,
                img_height,
            } => {
                if let Some(ref px) = pixels {
                    // Blit actual image
                    fb.draw_image(sx, sy, *img_width, *img_height, px);
                } else {
                    // Fallback: show alt text placeholder
                    // Background already drawn via bg_color
                    let tx = sx + b.style.padding_left as i32;
                    let ty = sy + b.style.padding_top as i32;
                    text_renderer.draw_text(
                        fb, alt, tx, ty, font_size, font_style,
                        color[0], color[1], color[2],
                    );
                }
            }

            LayoutContent::Block => {
                // Container — nothing to draw
            }
        }
    }
}
