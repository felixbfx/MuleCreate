"""Viewport presentation: native Metal on supported Macs, existing OpenGL elsewhere."""
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QSurfaceFormat

USE_METAL = False
if sys.platform == "darwin":
    from ..metal import wanted
    USE_METAL = wanted()

if USE_METAL:
    from PySide6.QtWidgets import QWidget as GPUWidget
else:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget as GPUWidget


def backend_name():
    return "metal" if USE_METAL else "gl"


class _GLBackend(GPUWidget):
    """QOpenGLWidget: OpenGL 3.3 compatibility profile (line widths > 1), moderngl on Qt's context."""

    def _setup_surface(self):
        fmt = QSurfaceFormat()
        if sys.platform == "darwin":
            fmt.setVersion(4, 1)
            fmt.setProfile(QSurfaceFormat.CoreProfile)
        else:
            fmt.setVersion(3, 3)
            fmt.setProfile(QSurfaceFormat.CompatibilityProfile)  # allows line widths > 1
        fmt.setDepthBufferSize(24)
        self.setFormat(fmt)  # the renderer does MSAA itself, Qt does not pass it through here

    def initializeGL(self):
        import moderngl
        from ..renderer import Renderer
        try:
            self.renderer = Renderer(moderngl.create_context(), self.context().functions())
        except Exception as exc:  # e.g. driver without OpenGL 3.3
            self.renderer, self._gl_error = None, str(exc)

    def paintGL(self):
        p = QPainter(self)
        if self.renderer is not None:
            # native GL between begin/endNativePainting, otherwise moderngl's GL state disturbs the QPainter
            p.beginNativePainting()
            self._paint_scene(self.renderer.ctx.detect_framebuffer(self.defaultFramebufferObject()))
            self._unbind_vao()
            p.endNativePainting()
        self._paint_overlay(p)
        p.end()

    def _unbind_vao(self):
        """moderngl leaves its last vertex array bound. Qt's QPainter would write its own
        attributes into it and destroy it -> always unbind after own GL."""
        self.context().extraFunctions().glBindVertexArray(0)

    def _with_gpu(self, fn):
        self.makeCurrent()
        try:
            return fn()
        finally:
            self._unbind_vao()
            self.doneCurrent()

    def request_frame(self):
        GPUWidget.update(self)

    def gpu_grab(self):
        return GPUWidget.grabFramebuffer(self)


class _MetalBackend(GPUWidget):
    """Qt input and layout around a MetalKit view; all scene GPU work uses native Metal 4."""

    def _setup_surface(self):
        self.setAttribute(Qt.WA_NativeWindow)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self._native_ctx = None
        self._native_attempted = False
        self._painting_native = False
        self._gl_error = None

    def _ensure_native(self):
        if self._native_ctx is not None:
            return
        if self._native_attempted:
            raise RuntimeError(self._gl_error or "GPU is not available")
        self._native_attempted = True
        ctx = None
        try:
            from ..metal.context import Context
            from ..renderer import Renderer
            ctx = Context(int(self.winId()))
            self.renderer = Renderer(ctx, ctx.gl)
            self._native_ctx = ctx
            self.destroyed.connect(lambda: ctx.release())
        except Exception as exc:
            if ctx is not None:
                ctx.release()
            self._gl_error = str(exc)
            self.renderer = None
            raise

    def paintEvent(self, event):
        try:
            self._render_native()
        except Exception:
            sys.excepthook(*sys.exc_info())
        if self.renderer is None:
            p = QPainter(self)
            try:
                self._paint_overlay(p)
            finally:
                p.end()

    def _render_native(self):
        if self._painting_native:
            return
        self._ensure_native()
        ctx = self._native_ctx
        dpr = self.devicePixelRatioF()
        size = (max(1, round(self.width() * dpr)), max(1, round(self.height() * dpr)))
        self._painting_native = True
        try:
            ctx.begin_frame(size)
            try:
                self._paint_scene(ctx.screen)
                overlay = getattr(self, "_overlay_image", None)
                if overlay is None or (overlay.width(), overlay.height()) != size:
                    overlay = self._overlay_image = QImage(*size, QImage.Format_RGBA8888_Premultiplied)
                overlay.setDevicePixelRatio(dpr)
                overlay.fill(0)
                painter = QPainter(overlay)
                try:
                    self._paint_overlay(painter)
                finally:
                    painter.end()
                ctx.present(overlay)
            finally:
                ctx.end_frame()
        finally:
            self._painting_native = False

    def _with_gpu(self, fn):
        self._ensure_native()
        ctx = self._native_ctx
        if ctx.in_frame():
            return fn()
        dpr = self.devicePixelRatioF()
        ctx.begin_frame((max(1, round(self.width() * dpr)), max(1, round(self.height() * dpr))))
        try:
            return fn()
        finally:
            ctx.end_frame()

    def request_frame(self):
        GPUWidget.update(self)

    def _unbind_vao(self):
        pass

    def makeCurrent(self):
        pass

    def doneCurrent(self):
        pass

    def grabFramebuffer(self):
        self._render_native()
        return self._native_ctx.grab()

    def gpu_grab(self):
        self._render_native()
        return self._native_ctx.grab()


GPUBackend = _MetalBackend if USE_METAL else _GLBackend
