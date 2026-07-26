function blendedColor = blendColor(colorA, colorB, blendFactor)
    blendedColor = blendFactor * colorA + (1 - blendFactor) * colorB;
end