'use client';

import { useEffect, useRef } from 'react';
import Image from 'next/image';

interface OptimizedImageProps {
  src: string;
  alt: string;
  width?: number;
  height?: number;
  priority?: boolean;
  className?: string;
  onLoad?: () => void;
  lazy?: boolean;
  placeholder?: 'blur' | 'empty';
  blurDataURL?: string;
}

export default function OptimizedImage({
  src,
  alt,
  width,
  height,
  priority = false,
  className,
  onLoad,
  lazy = true,
  placeholder = 'empty',
  blurDataURL,
}: OptimizedImageProps) {
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (!lazy || priority) return;

    // Use Intersection Observer for lazy loading
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const img = entry.target as HTMLImageElement;
            if (img.dataset.src) {
              img.src = img.dataset.src;
              img.onload = () => {
                img.classList.add('loaded');
                onLoad?.();
              };
              observer.unobserve(img);
            }
          }
        });
      },
      {
        rootMargin: '50px',
        threshold: 0.1,
      }
    );

    if (imgRef.current) {
      observer.observe(imgRef.current);
    }

    return () => observer.disconnect();
  }, [lazy, priority, onLoad]);

  // Generate srcSet for responsive images
  const generateSrcSet = (baseSrc: string) => {
    if (!baseSrc.includes('http') && !baseSrc.startsWith('/')) {
      return undefined;
    }

    const ext = baseSrc.split('.').pop();
    const baseName = baseSrc.replace(`.${ext}`, '');

    return [
      `${baseName}-400.${ext} 400w`,
      `${baseName}-800.${ext} 800w`,
      `${baseName}-1200.${ext} 1200w`,
      `${baseName}-1600.${ext} 1600w`,
    ].join(', ');
  };

  const sizes = width && height
    ? `(max-width: 768px) ${Math.min(width, 400)}px, (max-width: 1200px) ${Math.min(width, 800)}px, ${width}px`
    : '(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw';

  return (
    <Image
      ref={imgRef}
      src={src}
      alt={alt}
      width={width}
      height={height}
      priority={priority}
      className={className}
      onLoad={onLoad}
      placeholder={placeholder}
      blurDataURL={blurDataURL}
      sizes={sizes}
      style={{
        objectFit: 'cover',
        transition: 'opacity 0.3s ease-in-out',
      }}
    />
  );
}

// Utility for generating blur data URLs
export function generateBlurDataURL(width: number = 10, height: number = 10): string {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  // Create a simple gradient blur placeholder
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, '#f3f4f6');
  gradient.addColorStop(1, '#e5e7eb');

  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  return canvas.toDataURL();
}