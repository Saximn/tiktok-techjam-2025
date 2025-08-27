import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";

/**
 * FacePreview component
 * Displays whitelisted and non-whitelisted faces.
 * Non-whitelisted faces are blurred.
 */
type Face = {
  id: string;
  name: string;
  image: string;
  whitelisted: boolean;
};

interface FacePreviewProps {
  faces: Face[];
}

export function FacePreview({ faces }: FacePreviewProps) {
  return (
    <div className="flex gap-4 flex-wrap">
      {faces.map((face) => (
        <div key={face.id} className="flex flex-col items-center">
          <Avatar className="w-16 h-16">
            <AvatarImage
              src={face.image}
              alt={face.name}
              className={face.whitelisted ? "" : "blur-sm"}
            />
            <AvatarFallback>{face.name[0]}</AvatarFallback>
          </Avatar>
          <Badge
            variant={face.whitelisted ? "default" : "destructive"}
            className="mt-2"
          >
            {face.whitelisted ? "Whitelisted" : "Blurred"}
          </Badge>
          <span className="text-xs mt-1">{face.name}</span>
        </div>
      ))}
    </div>
  );
}
